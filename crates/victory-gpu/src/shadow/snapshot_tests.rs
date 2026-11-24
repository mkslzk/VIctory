use super::*;

/// The snapshots are one for all copies: one test at a time.
static ONE: Mutex<()> = Mutex::new(());

/// Parts kept of a copy, against a byte-by-byte model.
#[test]
fn snapshots_keep_each_declared_byte_once() {
    const LEN: usize = 300;
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut random = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    for _ in 0..2000 {
        let (mut parts, mut bytes, mut model) = (Vec::<Part>::new(), Vec::new(), [false; LEN]);
        for _ in 0..random(8) {
            let from = random(LEN);
            let to = from + random(LEN - from + 1);
            for (from, to) in uncovered(&parts, from, to) {
                let at = bytes.len();
                bytes.extend((from..to).map(|i| i as u8));
                parts.push(Part { from, to, at });
            }
            parts.sort_unstable_by_key(|p| p.from);
            model[from..to].fill(true);
        }
        assert_eq!(bytes.len(), model.iter().filter(|&&m| m).count());
        let mut room = Vec::new();
        for _ in 0..20 {
            let from = random(LEN);
            let to = from + 1 + random(LEN - from);
            let want: Option<Vec<u8>> = (from..to).map(|i| model[i].then_some(i as u8)).collect();
            assert_eq!(kept(&parts, &bytes, (from, to), &mut room), want.as_deref());
        }
    }
}

/// Two overlapping declared ranges and one apart: the bytes the GPU
/// changed go back, the title's own concurrent writes stay, and what
/// changed outside every declared range does not go back.
#[test]
fn copies_back_what_changed_in_declared_ranges() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    const LEN: usize = 8 * PAGE;
    let mut title = vec![0x11u8; LEN + PAGE];
    let start = (title.as_mut_ptr() as usize).next_multiple_of(PAGE);
    let Some((buffer, data)) = host().and_then(|h| h.shared_buffer(LEN as u64)) else {
        return; // no host GPU here
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Shadow::new(buffer, data, gpu, start as u64, LEN as u64);
    let ranges = [(100, 2 * PAGE), (PAGE, 3 * PAGE), (6 * PAGE + 8, 64)];
    assert_eq!(shadow.update(start, LEN), LEN);
    for &(at, len) in &ranges {
        shadow.writing(start + at, len);
    }
    let written = [150, PAGE + 5, 2 * PAGE, 4 * PAGE - 1, 6 * PAGE + 8];
    // SAFETY: inside the mapping and the title's memory.
    unsafe {
        for at in written.into_iter().chain([5 * PAGE]) {
            data.add(at).write(0x22);
        }
        ((start + 151) as *mut u8).write(0x33);
    }
    for &(at, len) in &ranges {
        shadow.copy_back(start + at, len);
    }
    for i in 0..LEN {
        // SAFETY: the title's memory.
        let byte = unsafe { ((start + i) as *const u8).read_volatile() };
        let want = match i {
            151 => 0x33,
            i if written.contains(&i) => 0x22,
            _ => 0x11,
        };
        assert_eq!(byte, want, "at {i:#x}");
    }
    assert!(shadow.before.lock().unwrap().is_empty());
    assert!(shadow.pending.lock().unwrap().is_empty());
    drop(title);
}

/// Two pages of title memory and a copy of it.
fn copy_of(title: &[u8]) -> Option<Shadow> {
    let (buffer, data) = host()?.shared_buffer(title.len() as u64)?;
    Some(Shadow::new(
        buffer,
        data,
        0,
        title.as_ptr() as u64,
        title.len() as u64,
    ))
}

/// How many parts and bytes are kept.
fn all_kept() -> (usize, usize) {
    let s = snapshots();
    (s.parts, s.bytes.len())
}

/// A copy dropped with ranges never copied back gives its snapshots up.
#[test]
fn a_dropped_copy_leaves_no_snapshot() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let title = vec![0x11u8; 2 * PAGE];
    let Some(a) = copy_of(&title) else {
        return; // no host GPU here
    };
    a.writing(title.as_ptr() as usize, 100);
    a.writing(title.as_ptr() as usize + PAGE, 300);
    assert_eq!(all_kept(), (2, 400));
    drop(a);
    assert_eq!(all_kept(), (0, 0));
}

/// Two copies' snapshots in one submission, taken in turns: each copies
/// back its own bytes, also after the other is gone.
#[test]
fn two_copies_keep_their_snapshots_apart() {
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let (first, second) = (vec![0x11u8; 2 * PAGE], vec![0x22u8; 2 * PAGE]);
    let (Some(a), Some(b)) = (copy_of(&first), copy_of(&second)) else {
        return; // no host GPU here
    };
    let (at, bt) = (first.as_ptr() as usize, second.as_ptr() as usize);
    assert_eq!(a.update(at, 2 * PAGE) + b.update(bt, 2 * PAGE), 4 * PAGE);
    a.writing(at, 100);
    b.writing(bt, 200);
    a.writing(at + PAGE, 300);
    b.writing(bt + PAGE, 400);
    assert_eq!(all_kept(), (4, 1000));
    // SAFETY: inside both mappings: what shaders wrote, one byte undeclared.
    unsafe {
        a.data.add(50).write(7);
        a.data.add(101).write(7);
        a.data.add(PAGE + 299).write(7);
        b.data.add(199).write(8);
        b.data.add(PAGE).write(8);
    }
    a.copy_back(at, 100);
    b.copy_back(bt, 200);
    a.copy_back(at + PAGE, 300);
    // `a` is done; `b`'s second range still has its bytes.
    assert_eq!(all_kept().0, 2);
    drop(a);
    assert_eq!(all_kept().0, 2);
    b.copy_back(bt + PAGE, 400);
    assert_eq!(all_kept(), (0, 0));
    let changed = |title: &[u8], was: u8| -> Vec<(usize, u8)> {
        let now = title.iter().enumerate();
        now.filter(|(_, v)| **v != was)
            .map(|(i, v)| (i, *v))
            .collect()
    };
    assert_eq!(
        changed(std::hint::black_box(&first), 0x11),
        [(50, 7), (PAGE + 299, 7)]
    );
    assert_eq!(
        changed(std::hint::black_box(&second), 0x22),
        [(199, 8), (PAGE, 8)]
    );
}
