//! Descriptor heaps: zeroed memory, with handles that are plain addresses
//! (CPU and GPU alike, as on the console).
//!
//! Descriptor sizes: views 32 bytes and samplers 16, the size of RDNA 2's
//! image and sampler descriptors, so a title that copies descriptors itself
//! moves whole ones. Render target and depth views are ours alone; 128 bytes
//! leaves room for whatever stride a title assumes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, DESCRIPTOR_HEAP, E_INVALIDARG, HRESULT, Iid, Object, Table, This, guid};
use crate::resource::{Memory, PAGEABLE_IIDS};

#[cfg(windows)]
pub(crate) mod host;

#[cfg(windows)]
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    kind: u32,
    count: u32,
    flags: u32,
    nodes: u32,
}

/// Newer titles (Grand Theft Auto VI) pass the description followed by
/// memory of their own for the descriptors: an address and its size in
/// bytes, or a null address to have the heap allocate (inferred from the
/// title's call sites; older titles pass the plain description).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Placement {
    address: u64,
    size: u64,
}

#[repr(C)]
pub(crate) struct Heap {
    object: Object,
    desc: HeapDesc,
    /// The first descriptor: in `memory`, or in the title's own memory.
    pub(crate) start: u64,
    memory: Option<Memory>,
    /// Shader-visible view and sampler heaps get a host heap with the same
    /// descriptor order when a draw first uses them (decision 0007, step
    /// 3); slots written before wait in `pending`.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<host::HostHeap>>,
    #[cfg(windows)]
    pending: std::sync::Mutex<Vec<u32>>,
    /// The write-watch asking (see [`crate::watch::writes`]) up to which
    /// the host heap has caught up with what the title wrote into its
    /// memory itself (see [`resync_title_writes`]).
    #[cfg(windows)]
    synced: std::sync::atomic::AtomicU64,
    /// The heap's memory for shaders that read its descriptors' dwords as
    /// numbers, made on first use (see [`Heap::bytes`]).
    #[cfg(windows)]
    copied: std::sync::OnceLock<Option<Bytes>>,
}

pub(crate) fn heap<'a>(this: This) -> &'a Heap {
    // SAFETY: called from descriptor heap methods only.
    unsafe { &*this.cast::<Heap>() }
}

impl Heap {
    /// Whether this is a sampler heap.
    pub(crate) fn samplers(&self) -> bool {
        self.desc.kind == SAMPLERS
    }

    /// The host heap, made on first use from what the title wrote so far.
    /// Made late because a host view heap the size of Grand Theft Auto VI's
    /// costs over 100 MB of commit, which its boot needs for its page pool.
    #[cfg(windows)]
    pub(crate) fn host(&self) -> Option<&host::HostHeap> {
        if let Some(h) = self.host.get() {
            return h.as_ref();
        }
        if self.desc.flags & SHADER_VISIBLE == 0 || !matches!(self.desc.kind, VIEWS_KIND | SAMPLERS)
        {
            return None;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // A recorder can take the fast path as soon as the OnceLock is
        // initialized. Publish only after replaying the queued writes.
        self.host
            .get_or_init(|| {
                let step = u64::from(increment(self.desc.kind)?);
                let h = host::HostHeap::new(self.samplers(), self.desc.count)?;
                for i in pending.drain(..) {
                    h.mirror(i, self.start + u64::from(i) * step, false);
                }
                Some(h)
            })
            .as_ref()
    }

    /// The descriptor index of `handle` in this heap, if it lies in it.
    pub(crate) fn index(&self, handle: u64) -> Option<u32> {
        let step = u64::from(increment(self.desc.kind)?);
        let at = handle.checked_sub(self.start)?;
        let index = at / step;
        (index < u64::from(self.desc.count)).then_some(index as u32)
    }

    /// The heap's memory in a host buffer, with its GPU address: what the
    /// window at `TITLE_SPACE` binds, so that a shader reading a
    /// descriptor's dwords as numbers reads the heap's bytes. The GPU
    /// cannot read the title's memory here, so this is a copy, made on
    /// first use and kept in step with every descriptor write through our
    /// methods (`record`, `copy_simple`) and with the title's own writes
    /// that [`resync_title_writes`] sees. A title that rewrites a slot in
    /// our own memory without a call of ours goes unseen, as its writes
    /// there already do for the host heap.
    #[cfg(windows)]
    pub(crate) fn bytes(&self) -> Option<(ID3D12Resource, u64)> {
        let step = u64::from(increment(self.desc.kind)?);
        let len = step * u64::from(self.desc.count.max(1));
        let copied = self.copied.get_or_init(|| {
            let (buffer, data) =
                crate::host::host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
            crate::host::name(&buffer, || {
                format!("descriptor heap {:#x}+{len:#x}", self.start)
            });
            // SAFETY: a live buffer, and `len` bytes at `start` (the heap's
            // memory: ours, or the title's in the placement it gave).
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            unsafe { std::ptr::copy_nonoverlapping(self.start as *const u8, data, len as usize) };
            let copied = Bytes {
                buffer,
                data,
                gpu,
                start: self.start,
                len,
            };
            // The heap's own bytes hold our tag where a view was recorded:
            // each such slot gets the descriptor's own dwords (see
            // [`dwords`]).
            for index in 0..self.desc.count {
                self.view_dwords(&copied, index, step);
            }
            Some(copied)
        });
        let copied = copied.as_ref()?;
        Some((copied.buffer.clone(), copied.gpu))
    }

    /// Copies the heap's slot `index` into the copy of its memory, if one
    /// was made: the descriptor changed.
    #[cfg(windows)]
    fn copy_slot(&self, index: u32, step: u64) {
        let Some(Some(copied)) = self.copied.get() else {
            return;
        };
        let at = u64::from(index) * step;
        if at + step > copied.len {
            return;
        }
        // SAFETY: both ranges inside live memory: the heap's, and the
        // copy's mapping. Writes to one slot race the title's own writes to
        // it, as our tags in that memory already do.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (copied.start + at) as *const u8,
                copied.data.add(at as usize),
                step as usize,
            )
        };
        // The copy followed the heap, tag and all: the descriptor's own
        // dwords follow (see [`dwords`]).
        self.view_dwords(copied, index, step);
    }

    /// Writes over the tag [`record`] left in a slot's copy the descriptor
    /// dwords the slot's view determines (see [`dwords`]), so that a shader
    /// reading them as numbers reads the descriptor and not our bookkeeping.
    /// A slot no view names keeps the heap's own bytes.
    #[cfg(windows)]
    fn view_dwords(&self, copied: &Bytes, index: u32, step: u64) {
        let at = u64::from(index) * step;
        if at + step > copied.len {
            return;
        }
        // SAFETY: the slot lies in the heap's memory, whose first dword is
        // our tag when a view was recorded there.
        let tag = unsafe { ((copied.start + at) as *const u32).read_unaligned() };
        if tag != VIEW_TAG {
            return;
        }
        // The lookup is the one `view` makes for a handle: the slot's own
        // address.
        let Some(view) = view(copied.start + at) else {
            return;
        };
        let Some(dwords) = dwords(&view) else {
            return;
        };
        // SAFETY: `at + step` lies inside the copy's mapping (checked
        // above).
        let slot =
            unsafe { std::slice::from_raw_parts_mut(copied.data.add(at as usize), step as usize) };
        write_dwords(slot, &dwords);
    }
}

/// A copy of a shader-visible heap's memory (see [`Heap::bytes`]).
#[cfg(windows)]
struct Bytes {
    buffer: ID3D12Resource,
    /// The buffer's mapping: written with the heap's bytes.
    data: *mut u8,
    /// The buffer's GPU address.
    gpu: u64,
    /// Where the copy starts in the heap, and how long it is.
    start: u64,
    len: u64,
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only for slots that change.
#[cfg(windows)]
unsafe impl Send for Bytes {}
// SAFETY: as above.
#[cfg(windows)]
unsafe impl Sync for Bytes {}

/// The GFX10 descriptor dwords a [`View`] determines, for the copy of a
/// heap's memory a shader reads as numbers (see [`Heap::bytes`]): the tag
/// [`record`] leaves in a slot's first three dwords is not the descriptor.
///
/// The layout is the translation's own (`victory-shader`'s
/// `emit/src/lower.rs`, `descriptor_data` and `descriptor_stride`):
///
/// * the base address: dword 0, and dword 1's bits 15:0;
/// * a buffer's stride: dword 1's bits 29:16 (bits 31:30 are the cache
///   swizzle, left zero), its record count: dword 2;
/// * a texture's width - 1: dword 1's bits 31:30 and dword 2's 0-13 (one
///   field across them), height - 1: dword 2's 14-29, identity swizzle,
///   type and last level: dword 3, depth or layers - 1: dword 4.
///
/// Nothing else is the view's: a descriptor's format words, its tiling
/// bits, the fields of a constant buffer's size and every sampler field
/// keep the heap's own bytes, and so does a texture view of a kind the
/// translation does not name (MSAA, cube) and one naming part of the mip
/// chain. `None` for a view that says nothing about its descriptor's
/// dwords.
fn dwords(view: &View) -> Option<Dwords> {
    let (address, tail) = match *view {
        View::Constant(address, _size) => (address, Tail::None),
        View::Shader(resource, ref desc) => named(resource as This, desc.as_ref())?,
        View::Unordered(resource, _, ref desc) => named(resource as This, desc.as_ref())?,
        View::Placed(_, ref resource, ref desc) => placed(resource, desc)?,
        View::Sampler(_) | View::ConsoleSampler(_) => return None,
    };
    Some(Dwords { address, tail })
}

/// A descriptor's dwords (see [`dwords`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Dwords {
    /// Its base address: dword 0, and dword 1's bits 15:0.
    address: u64,
    tail: Tail,
}

/// The dwords a descriptor holds past its address (see [`dwords`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tail {
    /// Only the address: a constant buffer's size is in no field this view
    /// names.
    None,
    /// A buffer's: its stride, in dword 1's bits 29:16, and its record
    /// count, in dword 2.
    Buffer { stride: u32, records: u32 },
    /// A texture's dimensions, in dwords 2 to 4; dword 3 (the swizzle, the
    /// type and the last level) only when the view names the type and the
    /// whole mip chain.
    Image {
        width: u32,
        height: u32,
        depth: u32,
        /// The T# type and the last level.
        levels: Option<(u32, u32)>,
    },
}

/// The dwords a shader or unordered access view of `resource` determines.
fn named(resource: This, words: Option<&[u32; 10]>) -> Option<(u64, Tail)> {
    if let Some((data, bytes)) = crate::resource::buffer_bytes(resource) {
        return Some((data as u64, buffer_tail(bytes, words)));
    }
    let (data, desc) = crate::resource::texture_memory(resource)?;
    let described = crate::resource::described(&desc);
    Some((data as u64, image_tail(&described, words)))
}

/// The dwords a view of memory without a resource object determines: the
/// address its description names (dwords 10 and 11) and the fields of its
/// resource and view descriptions.
fn placed(resource: &[u32; 14], desc: &[u32; 16]) -> Option<(u64, Tail)> {
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let described = crate::resource::described_words(resource);
    let words: &[u32; 10] = desc[..10].try_into().ok()?;
    Some((
        address,
        if described.buffer() {
            // The words' width is a buffer's bytes (see `buffer_range`).
            buffer_tail(described.width, Some(words))
        } else {
            image_tail(&described, Some(words))
        },
    ))
}

/// A buffer view's tail: the record count its description names, else what
/// the buffer holds, and the stride that scales a record — the view's
/// structure stride, else its format's element (a typed view's record is
/// one format unit), else a dword: the raw form a view naming neither (its
/// format is UNKNOWN, the raw flag's form) is read by.
fn buffer_tail(bytes: u64, words: Option<&[u32; 10]>) -> Tail {
    // A buffer view's description: first element, record count, structure
    // stride and flags, in dwords 2 to 5.
    let stride = words.map_or(0, |w| if w[4] > 0 { w[4] } else { element_bytes(w[0]) });
    let stride = if stride > 0 { stride } else { 4 };
    let declared = words.map_or(0, |w| w[3]);
    let records = if declared > 0 {
        declared
    } else {
        u32::try_from(bytes / u64::from(stride)).unwrap_or(u32::MAX)
    };
    Tail::Buffer { stride, records }
}

/// A texture view's tail: the resource's dimensions and mip chain, and the
/// type of image its view dimension names (8 1D, 9 2D, 10 3D, 12 1D array,
/// 13 2D array — the translation's table — for MSAA and cube ones it names
/// none).
fn image_tail(described: &crate::resource::Described, words: Option<&[u32; 10]>) -> Tail {
    // A texture view's description: format, dimension, then the most
    // detailed mip and the level count, in dwords 2 and 3.
    let (kind, whole) = match words {
        Some(w) => {
            // D3D12_SRV/UAV_DIMENSION: 2 1D, 3 1D array, 4 2D, 5 2D array,
            // 8 3D.
            let kind = match w[1] {
                2 => 8,
                4 => 9,
                8 => 10,
                3 => 12,
                5 => 13,
                _ => 0,
            };
            let levels = described.mip_levels;
            (kind, w[2] == 0 && (w[3] == 0 || w[3] == levels))
        }
        // The resource's own view: a plain 1D, 2D or 3D one (an array's is
        // not named), of its whole chain.
        None => (
            match described.dimension {
                2 => 8,
                3 if described.depth_or_array_size <= 1 => 9,
                4 => 10,
                _ => 0,
            },
            true,
        ),
    };
    let levels = (whole && kind != 0).then(|| (kind, described.mip_levels.max(1) - 1));
    Tail::Image {
        width: u32::try_from(described.width).unwrap_or(u32::MAX),
        height: described.height,
        depth: described.depth_or_array_size,
        levels,
    }
}

/// A view's format as bytes per element; 0 for a format without one (a
/// raw view's, or one [`crate::resource::element`] does not know).
fn element_bytes(format: u32) -> u32 {
    crate::resource::element(crate::resource::pc_format(format))
        .map_or(0, |(bytes, _)| bytes as u32)
}

/// Writes `dwords` over `slot`, a copy of a descriptor slot.
fn write_dwords(slot: &mut [u8], dwords: &Dwords) {
    let mut put = |at: usize, word: u32| {
        if let Some(dst) = slot.get_mut(at..at + 4) {
            dst.copy_from_slice(&word.to_le_bytes());
        }
    };
    let hi = (dwords.address >> 32) as u32 & 0xffff;
    put(0, dwords.address as u32);
    match dwords.tail {
        Tail::None => put(4, hi),
        Tail::Buffer { stride, records } => {
            put(4, hi | (stride & 0x3fff) << 16);
            put(8, records);
        }
        Tail::Image {
            width,
            height,
            depth,
            levels,
        } => {
            let (w, h) = (width.max(1) - 1, height.max(1) - 1);
            put(4, hi | (w & 3) << 30);
            put(8, (w >> 2) & 0x3fff | (h & 0xffff) << 14);
            if let Some((kind, last)) = levels {
                put(12, 0xfac | kind << 28 | (last & 0xff) << 16);
            }
            put(16, depth.max(1) - 1);
        }
    }
}

const VIEWS_KIND: u32 = 0;
const SAMPLERS: u32 = 1;
const SHADER_VISIBLE: u32 = 1;

/// Shader-visible heaps by address range, for mirroring descriptor writes:
/// start, end, heap.
#[cfg(windows)]
static SHADER_HEAPS: std::sync::Mutex<Vec<(u64, u64, usize)>> = std::sync::Mutex::new(Vec::new());

/// The shader-visible heap holding `handle`, and the descriptor's index.
#[cfg(windows)]
fn shader_slot<'a>(handle: u64) -> Option<(&'a Heap, u32)> {
    let heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let &(_, _, at) = heaps
        .iter()
        .find(|&&(start, end, _)| (start..end).contains(&handle))?;
    // SAFETY: registered heaps are live (removed when dropped).
    let heap = unsafe { &*(at as *const Heap) };
    Some((heap, heap.index(handle)?))
}

/// Makes the host descriptor for what the title wrote at `handle`, if it
/// is in a shader-visible heap.
#[cfg(windows)]
fn mirror(handle: u64) {
    let Some((heap, index)) = shader_slot(handle) else {
        return;
    };
    // The descriptor's bytes changed: its heap's copy of them follows.
    if let Some(step) = increment(heap.desc.kind) {
        heap.copy_slot(index, u64::from(step));
    }
    let mut pending = heap.pending.lock().unwrap_or_else(|e| e.into_inner());
    match heap.host.get() {
        None => pending.push(index),
        Some(host) => {
            drop(pending);
            if let Some(host) = host {
                host.mirror(index, handle, false);
            }
        }
    }
}

/// Brings the host heaps up to date with descriptors the title wrote into
/// heap memory itself (copies of whole slots, which no create or copy call
/// reports), before a submission runs: slots read directly by index (not
/// through a table) are made only from what was mirrored, so a slot the
/// title turned from a texture into a constant buffer kept the texture,
/// whose upper half the shader read as a buffer at address 0: a GPU page
/// fault and a lost device in the world. Only the 64 KiB blocks written
/// since the last call are looked at; heaps in memory the kernel does not
/// watch (ours, not the title's) are left as they are.
#[cfg(windows)]
pub(crate) fn resync_title_writes() {
    use std::sync::atomic::Ordering::Relaxed;
    // Held throughout: registered heaps stay alive while it is.
    let heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    for &(start, end, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        let (Some(Some(host)), Some(step)) = (heap.host.get(), increment(heap.desc.kind)) else {
            continue; // no host heap yet: its first use mirrors everything
        };
        let page = crate::watch::PAGE as u64;
        let (first, last) = (start / page * page, end.div_ceil(page) * page);
        let Some((now, written)) = crate::watch::writes(first as usize, last as usize, true) else {
            continue;
        };
        let before = heap.synced.swap(now, Relaxed);
        let step = u64::from(step);
        for (p, asked) in written {
            if asked <= before {
                continue;
            }
            let from = (p as u64 * page).max(start);
            let to = ((p as u64 + 1) * page).min(end);
            for i in (from - start) / step..(to - start).div_ceil(step) {
                let i = i as u32;
                if i < heap.desc.count {
                    host.mirror(i, start + u64::from(i) * step, false);
                    // What the shader reads as a number follows too.
                    heap.copy_slot(i, step);
                }
            }
        }
    }
}

/// Gives back dynamic slots of the host heap behind the title heap at
/// `heap`, if that heap still exists.
#[cfg(windows)]
pub(crate) fn release_dynamic(heap: usize, first: u32, n: u32) {
    let heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.release(first, n);
    }
}

#[cfg(windows)]
impl Drop for Heap {
    fn drop(&mut self) {
        let at = std::ptr::from_ref::<Heap>(self) as usize;
        let mut heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        heaps.retain(|&(_, _, h)| h != at);
    }
}

/// Bytes per descriptor, by heap type (views, samplers, render targets,
/// depth).
fn increment(kind: u32) -> Option<u32> {
    [32, 16, 128, 128].get(kind as usize).copied()
}

/// Device method.
pub(crate) extern "system" fn increment_size(_: This, kind: u32) -> u32 {
    increment(kind).unwrap_or(0)
}

/// What our render target views hold: the resource and the view
/// description, turned into a host view when a command uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RtvRecord {
    tag: u32,
    described: u32,
    pub(crate) resource: This,
    desc: [u32; 6],
}
const RTV_TAG: u32 = u32::from_le_bytes(*b"vRTV");
const _: () = assert!(size_of::<RtvRecord>() <= 112);

impl RtvRecord {
    /// The view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC> {
        const TEXTURE_2D: u32 = 4;
        self.host_desc(TEXTURE_2D)
    }

    /// The depth view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn depth_desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_VIEW_DESC> {
        const TEXTURE_2D: u32 = 3;
        self.host_desc(TEXTURE_2D)
    }

    /// The description words as a host type, the format as the host has it.
    /// Without one, a typeless host copy gets a 2D view (`dimension`) in
    /// the title's format, which the host needs spelled out.
    #[cfg(windows)]
    fn host_desc<T>(&self, dimension: u32) -> Option<T> {
        assert!(size_of::<T>() <= size_of::<[u32; 6]>());
        let mut words = if self.described != 0 {
            self.desc
        } else {
            [
                crate::resource::view_format(self.resource)?,
                dimension,
                0,
                0,
                0,
                0,
            ]
        };
        words[0] = crate::resource::pc_format(words[0]);
        // SAFETY: both host view descriptions are 24 bytes in this layout.
        Some(unsafe { (&raw const words).cast::<T>().read_unaligned() })
    }
}

/// Device method: records the view at `handle` (descriptor memory).
pub(crate) extern "system" fn create_rtv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    if handle == 0 {
        return;
    }
    // SAFETY: the title passes a description or null.
    let described = unsafe { desc.as_ref() };
    let record = RtvRecord {
        tag: RTV_TAG,
        described: u32::from(described.is_some()),
        resource,
        desc: described.copied().unwrap_or_default(),
    };
    // SAFETY: the handle is a descriptor slot, at least 112 bytes.
    unsafe { (handle as *mut RtvRecord).write_unaligned(record) };
}

/// Device method: a depth view, recorded as render target views are.
pub(crate) extern "system" fn create_dsv(
    this: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    create_rtv(this, resource, desc, handle);
    if handle != 0 {
        // SAFETY: just written by `create_rtv`.
        unsafe { (handle as *mut u32).write_unaligned(DSV_TAG) };
    }
}
const DSV_TAG: u32 = u32::from_le_bytes(*b"vDSV");

/// A shader view or sampler as the title described it (descriptions in the
/// host's layout, kept as words), for the host to build its own from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum View {
    /// Constant buffer: GPU address and size.
    Constant(u64, u32),
    /// Shader resource: resource, description (none: the resource's own).
    Shader(usize, Option<[u32; 10]>),
    /// Unordered access: resource, counter resource, description.
    Unordered(usize, usize, Option<[u32; 10]>),
    Sampler([u32; 13]),
    /// The console's own sampler description: flags, filters per axis,
    /// addressing, LOD and anisotropy settings, border color, then
    /// console-only tuning words.
    ConsoleSampler([u32; 27]),
    /// A view of memory without a resource object: whether it is unordered
    /// access, the resource description, then the view's (the plain one
    /// followed by the memory address and console fields).
    Placed(bool, [u32; 14], [u32; 16]),
}

impl View {
    /// Whether the view reads buffer memory (constant buffers, buffer views
    /// of resources or of memory) or memory the title placed a texture in:
    /// such views get host slots only for draws.
    #[cfg(windows)]
    fn is_buffer(&self) -> bool {
        match self {
            View::Constant(..) | View::Placed(..) => true,
            View::Shader(r, _) | View::Unordered(r, _, _) => {
                crate::resource::buffer_bytes(*r as This).is_some()
            }
            View::Sampler(_) | View::ConsoleSampler(_) => false,
        }
    }
}

/// Every distinct view in use; a descriptor holds a tag, an index into it
/// and the entry's generation, which survive titles copying descriptors as
/// bytes. A view of a resource lives as long as the resource: when the
/// title releases it, its entries are freed for reuse under a new
/// generation, so descriptors (and host slots made from them) naming the
/// old one go stale instead of naming whatever is made at the same address
/// next.
/// shortcut: views without a resource object (constant buffers, memory
/// views, samplers) are kept for the run; they are bounded by the title's
/// memory and sampler states.
struct Views {
    /// Each entry's view and generation.
    list: Vec<(View, u32)>,
    index: std::collections::BTreeMap<View, u32>,
    /// Entries naming each resource.
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
}

static VIEWS: std::sync::Mutex<Views> = std::sync::Mutex::new(Views {
    list: Vec::new(),
    index: std::collections::BTreeMap::new(),
    by_resource: std::collections::BTreeMap::new(),
    free: Vec::new(),
});
const VIEW_TAG: u32 = u32::from_le_bytes(*b"vVIE");

impl View {
    /// The resource objects the view names.
    fn resources(&self) -> [usize; 2] {
        match *self {
            View::Shader(r, _) => [r, 0],
            View::Unordered(r, counter, _) => [r, counter],
            _ => [0, 0],
        }
    }
}

fn record(handle: u64, view: View) {
    if handle == 0 {
        return;
    }
    let mut views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let i = match views.index.get(&view) {
        Some(&i) => i,
        None => {
            let i = match views.free.pop() {
                Some(i) => {
                    views.list[i as usize].0 = view.clone();
                    i
                }
                None => {
                    views.list.push((view.clone(), 1));
                    views.list.len() as u32 - 1
                }
            };
            for r in view.resources().into_iter().filter(|&r| r != 0) {
                views.by_resource.entry(r).or_default().push(i);
            }
            views.index.insert(view, i);
            i
        }
    };
    let generation = views.list[i as usize].1;
    // SAFETY: the handle is a descriptor slot, at least 16 bytes.
    unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
    drop(views);
    #[cfg(windows)]
    mirror(handle);
}

/// The title released `resource`: the views naming it go stale.
pub(crate) fn forget(resource: usize) {
    let mut views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(entries) = views.by_resource.remove(&resource) else {
        return;
    };
    for i in entries {
        let (view, generation) = views.list[i as usize].clone();
        // Entries freed through a view's other resource may hold another
        // view by now.
        if views.index.get(&view) != Some(&i) || !view.resources().contains(&resource) {
            continue;
        }
        views.index.remove(&view);
        views.list[i as usize].1 = generation.wrapping_add(1).max(1);
        // The view's other resource (a counter) keeps a stale entry, which
        // `forget` of that resource skips.
        views.free.push(i);
    }
}

/// The view of entry `i` if `generation` is still its own.
#[cfg(windows)]
fn view_at(i: u32, generation: u32) -> Option<View> {
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let (view, current) = views.list.get(i as usize)?;
    (*current == generation).then(|| view.clone())
}

/// The view recorded at `handle`, if one is (and its resource is alive).
pub(crate) fn view(handle: u64) -> Option<View> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let (view, current) = views.list.get(i as usize)?;
    (tag == VIEW_TAG && *current == generation).then(|| view.clone())
}

/// A description of `N` words, if the title passes one.
fn words<const N: usize>(desc: *const [u32; N]) -> Option<[u32; N]> {
    // SAFETY: the title passes a description or null.
    (!desc.is_null()).then(|| unsafe { desc.read_unaligned() })
}

/// Device method.
pub(crate) extern "system" fn create_cbv(_: This, desc: *const [u32; 4], handle: u64) {
    if let Some([lo, hi, size, _]) = words(desc) {
        let address = u64::from(lo) | u64::from(hi) << 32;
        record(handle, View::Constant(address, size));
    }
}

/// Device method.
pub(crate) extern "system" fn create_srv(
    _: This,
    resource: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    record(handle, View::Shader(resource as usize, words(desc)));
}

/// Device method.
pub(crate) extern "system" fn create_uav(
    _: This,
    resource: This,
    counter: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    let view = View::Unordered(resource as usize, counter as usize, words(desc));
    record(handle, view);
}

/// Device methods 0x230 and 0x238: shader resource and unordered access
/// views of memory the title placed itself, `(resource desc, view desc,
/// handle)`.
pub(crate) extern "system" fn create_placed_srv(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(false, resource, desc));
    }
}

pub(crate) extern "system" fn create_placed_uav(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(true, resource, desc));
    }
}

/// Device method.
pub(crate) extern "system" fn create_sampler(_: This, desc: *const [u32; 13], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::Sampler(desc));
    }
}

/// Device method 0x1e8.
pub(crate) extern "system" fn create_sampler_x(_: This, desc: *const [u32; 27], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::ConsoleSampler(desc));
    }
}

/// Device method: `count` descriptors of `kind` from `src` to `dst`.
pub(crate) extern "system" fn copy_simple(_: This, count: u32, dst: u64, src: u64, kind: u32) {
    let step = u64::from(increment(kind).unwrap_or(0));
    let len = u64::from(count) * step;
    if dst != 0 && src != 0 {
        // SAFETY: both ranges are descriptor slots the title owns.
        unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len as usize) };
        #[cfg(windows)]
        for i in 0..u64::from(count) {
            mirror(dst + i * step);
        }
    }
}

/// Device method: ranges of descriptors (start and count each) gathered
/// from the sources and scattered to the destinations, in order.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn copy(
    _: This,
    dst_ranges: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    src_ranges: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    kind: u32,
) {
    let Some(step) = increment(kind) else {
        return;
    };
    // Each range as (start, count); a null size array means one descriptor
    // per range.
    let ranges = |n: u32, starts: *const u64, sizes: *const u32| -> Vec<(u64, u32)> {
        (0..n as usize)
            .map(|i| {
                // SAFETY: the title passes `n` starts and sizes (or null sizes).
                unsafe {
                    (
                        *starts.add(i),
                        if sizes.is_null() { 1 } else { *sizes.add(i) },
                    )
                }
            })
            .collect()
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let src = ranges(src_ranges, src_starts, src_sizes);
    let mut dst = ranges(dst_ranges, dst_starts, dst_sizes).into_iter();
    let (mut to, mut room) = dst.next().unwrap_or((0, 0));
    for (mut from, mut left) in src {
        while left > 0 {
            while room == 0 {
                let Some(next) = dst.next() else { return };
                (to, room) = next;
            }
            let n = left.min(room);
            copy_simple(std::ptr::null(), n, to, from, kind);
            let bytes = u64::from(n) * u64::from(step);
            (to, from, room, left) = (to + bytes, from + bytes, room - n, left - n);
        }
    }
}

/// The view recorded at `handle`, if one is.
pub(crate) fn rtv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, RTV_TAG)
}

/// The depth view recorded at `handle`, if one is.
pub(crate) fn dsv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, DSV_TAG)
}

fn tagged_record(handle: u64, tag: u32) -> Option<RtvRecord> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let record = unsafe { (handle as *const RtvRecord).read_unaligned() };
    (record.tag == tag).then_some(record)
}

/// Device method.
pub(crate) extern "system" fn create_heap(
    _: This,
    desc_ptr: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc_ptr.as_ref() }) else {
        return E_INVALIDARG;
    };
    let Some(size) = increment(desc.kind).map(|i| u64::from(i) * u64::from(desc.count.max(1)))
    else {
        return E_INVALIDARG;
    };
    // SAFETY: read only as a hint: a title that passes the plain
    // description has its own stack or data after it.
    let placement = unsafe { desc_ptr.add(1).cast::<Placement>().read_unaligned() };
    // shortcut: a placement is recognised by its size matching the count,
    // since the plain and extended descriptions share the slot and the
    // interface version; find the flag that tells them apart if a title
    // trips over this.
    let placed = placement.address != 0 && placement.size == size;
    tracing::debug!(?desc, placed, "descriptor heap");
    let (start, memory) = if placed {
        (placement.address, None)
    } else {
        match Memory::zeroed(size, 64) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    };
    let visible = desc.flags & SHADER_VISIBLE != 0 && matches!(desc.kind, VIEWS_KIND | SAMPLERS);
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        start,
        memory,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        pending: Default::default(),
        #[cfg(windows)]
        synced: std::sync::atomic::AtomicU64::new(0),
        #[cfg(windows)]
        copied: std::sync::OnceLock::new(),
    });
    #[cfg(windows)]
    if visible {
        let at = std::ptr::from_ref::<Heap>(&heap) as usize;
        let mut heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        heaps.push((start, start + size, at));
    }
    #[cfg(not(windows))]
    let _ = visible;
    object::hand_out(DESCRIPTOR_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &HEAP_IIDS).then_some(this);
    object::answer(DESCRIPTOR_HEAP, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(heap(this).desc) };
    }
}
/// The same address for the CPU and the GPU start.
extern "system" fn start(this: This) -> u64 {
    heap(this).start
}

/// IUnknown 3, object 4, device child 1, heap 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<DESCRIPTOR_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = desc as *const (); // 0x40
    f[9] = start as *const (); // 0x48 CPU
    f[10] = start as *const (); // 0x50 GPU
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{S_OK, slot};

    /// What newer titles pass; tests use it throughout, so that reading the
    /// placement stays inside the value.
    #[repr(C)]
    struct Extended(HeapDesc, Placement);

    fn create(desc: HeapDesc, placement: Placement, out: &mut This) -> HRESULT {
        let extended = Extended(desc, placement);
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], out)
    }

    #[test]
    fn heaps_hold_count_descriptors_from_their_start() {
        let null = std::ptr::null();
        let desc = HeapDesc {
            kind: 0,
            count: 1000,
            flags: 1,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = null;
        assert_eq!(create(desc, none, &mut h), S_OK);
        let cpu = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);
        assert_eq!(cpu % 64, 0);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), cpu);
        assert_eq!(increment_size(null, 0), 32);
        // SAFETY: the heap holds 1000 descriptors of 32 bytes.
        unsafe { (cpu as *mut u8).add(999 * 32 + 31).write(1) };
        assert_eq!(increment_size(null, 9), 0);
        let bad = HeapDesc { kind: 9, ..desc };
        assert_eq!(create(bad, none, &mut h), E_INVALIDARG);
    }

    /// A slot's copy holds the descriptor's dwords, not our tag: the
    /// address, a buffer's stride and record count, a texture's size and
    /// type. What the view does not name keeps the heap's bytes.
    #[test]
    fn a_slots_copy_holds_the_descriptors_dwords() {
        let words = |view: &View| -> [u32; 5] {
            let mut slot = [0xaau8; 32];
            write_dwords(&mut slot, &dwords(view).expect("a descriptor's dwords"));
            let mut out = [0u32; 5];
            for (i, w) in out.iter_mut().enumerate() {
                *w = u32::from_le_bytes(slot[4 * i..4 * i + 4].try_into().expect("a dword"));
            }
            out
        };
        // A constant buffer's address, and only that.
        let constant = words(&View::Constant(0x7f00_0000_0100, 256));
        assert_eq!(&constant[..2], &[0x100, 0x7f00]);
        assert_eq!(&constant[2..], &[0xaaaa_aaaa; 3], "the size is no dword");
        // A view's words: a description of ten, then the address in dwords
        // 10 and 11.
        let at = |address: u64| {
            let mut view = [0u32; 16];
            view[10] = address as u32;
            view[11] = (address >> 32) as u32;
            view
        };
        // A structured view of memory: 100 records of 16 bytes at
        // 0x7f00_1234_5600, over a buffer description of 0x1000 bytes.
        let resource = [1, 0, 0, 0, 0x1000, 0, 1, 1, 0, 1, 0, 0, 0, 0];
        let mut view = at(0x7f00_1234_5600);
        view[1] = 1; // a buffer
        view[3] = 100; // its record count
        view[4] = 16; // its structure stride
        let structured = words(&View::Placed(false, resource, view));
        assert_eq!(structured[0], 0x1234_5600);
        assert_eq!(structured[1], 0x7f00 | 16 << 16, "the stride");
        assert_eq!(structured[2], 100, "the record count");
        assert_eq!(
            &structured[3..],
            &[0xaaaa_aaaa; 2],
            "the format words are not the view's"
        );
        // A 2D texture of 1024 x 512: the width - 1 spans dword 1's top two
        // bits and dword 2's low 14, and dword 3 has the type and last level.
        let resource = [3, 0, 0, 0, 1024, 0, 512, 1, 28, 1, 0, 0, 0, 0];
        let mut view = at(0x1000);
        view[0] = 28; // its format
        view[1] = 4; // a 2D view
        let texture = words(&View::Placed(false, resource, view));
        assert_eq!(texture[0], 0x1000);
        assert_eq!(texture[1], 3 << 30);
        assert_eq!(texture[2], 0xff | 511 << 14);
        assert_eq!(texture[3], 0xfac | 9 << 28);
        // A sampler's four dwords are not a descriptor this can rebuild.
        assert_eq!(dwords(&View::Sampler([0; 13])), None);
        assert_eq!(dwords(&View::ConsoleSampler([0; 27])), None);
    }

    #[test]
    fn views_survive_copies() {
        let null = std::ptr::null();
        let mut heap = vec![0u64; 4 * 8]; // four 32-byte slots
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let (s0, s1, s2, s3) = (at(0), at(1), at(2), at(3));
        create_cbv(null, &[0x1000, 1, 256, 0], s0);
        assert_eq!(view(s0), Some(View::Constant(0x1_0000_1000, 256)));
        create_srv(null, null, std::ptr::null(), s1);
        assert_eq!(view(s1), Some(View::Shader(0, None)));
        // The same view twice is one entry.
        create_srv(null, null, std::ptr::null(), s2);
        // SAFETY: both slots are in `heap`.
        assert_eq!(unsafe { *(s1 as *const [u32; 3]) }, unsafe {
            *(s2 as *const [u32; 3])
        });
        // Two ranges of one into one range of two.
        let (dst, dst_n, src, src_n) = ([s2], [2u32], [s0, s1], [1u32, 1]);
        copy(
            null,
            1,
            dst.as_ptr(),
            dst_n.as_ptr(),
            2,
            src.as_ptr(),
            src_n.as_ptr(),
            0,
        );
        assert_eq!(view(s2), view(s0));
        assert_eq!(view(s3), view(s1));
        assert_eq!(view(at(0) + 8), None);
        heap[0] = 0;
    }

    /// A released resource's views go stale; the same view of a new
    /// resource at the same address is a new record.
    #[test]
    fn views_die_with_their_resource() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let (a, b) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 32);
        let resource = 0x7777_0000usize as This;
        create_srv(null, resource, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Shader(0x7777_0000, None)));
        forget(0x7777_0000);
        assert_eq!(view(a), None, "stale after release");
        create_srv(null, resource, std::ptr::null(), b);
        assert_eq!(view(b), Some(View::Shader(0x7777_0000, None)));
        // SAFETY: both slots are in `slots`.
        let (old, new) = unsafe { (*(a as *const [u32; 3]), *(b as *const [u32; 3])) };
        assert_ne!(old, new, "a new generation");
    }

    /// Grand Theft Auto VI places its shader-visible heap in its own memory
    /// and checks that the heap starts there.
    #[test]
    fn placed_heaps_start_at_the_title_memory() {
        let mut memory = vec![0u8; 64 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        assert_eq!(
            create(
                desc,
                Placement {
                    address,
                    size: 64 * 32
                },
                &mut h
            ),
            S_OK
        );
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), address);
        // A size that does not match the count is not a placement.
        assert_eq!(create(desc, Placement { address, size: 7 }, &mut h), S_OK);
        assert_ne!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
    }
}
