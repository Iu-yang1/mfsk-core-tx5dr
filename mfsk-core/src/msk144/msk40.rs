//! MSK40 shorthand: 40 ms burst framing and (32,16) LDPC, independent of standard MSK144.
//! WSJT-X references: genmsk40.f90, encode_msk40.f90, nhash.c.
//!
//! IMPORTANT: known-calls offline matched-waveform decoding is not the full
//! WSJT-X msk40spd/bpdecode40 receiver. Do not route this to physical PTT
//! or claim MSK40 P9 support without full decoder/equalizer acceptance.

use alloc::{format, string::String, vec, vec::Vec};
#[cfg(any(feature = "fft-rustfft", feature = "fft-extern"))]
use num_complex::Complex32;
#[cfg(not(feature = "std"))]
use num_traits::Float;

const FS: usize = 12_000;
const NSPS: usize = 6;
const FRAME_TONES: usize = 40;
const FRAME_SAMPLES: usize = FRAME_TONES * NSPS;
const GENERATOR: [u16; 16] = [
    0x4428, 0x5a6b, 0x1b04, 0x2c12, 0x60c4, 0x1071, 0xbe6a, 0x36dd, 0xc580, 0xad9a, 0xeca2, 0x7843,
    0x332e, 0xa685, 0x5906, 0x1efe,
];
const COLORDER: [usize; 32] = [
    4, 1, 2, 3, 0, 8, 6, 10, 13, 28, 20, 23, 17, 15, 27, 25, 16, 12, 18, 19, 7, 21, 22, 11, 24, 5,
    26, 14, 9, 29, 30, 31,
];
const SYNC: [u8; 8] = [1, 0, 1, 1, 0, 0, 0, 1];
const REPORTS: [&str; 16] = [
    "-03", "+00", "+03", "+06", "+10", "+13", "+16", "R-03", "R+00", "R+03", "R+06", "R+10",
    "R+13", "R+16", "RRR", "73",
];

// WSJT-X hash(string, 37, init=146) uses Bob Jenkins lookup3 and masks
// its result to 12 bits; FT8 ihashcall uses a DIFFERENT algorithm.
fn mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(4);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(6);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(8);
    *b = b.wrapping_add(*a);
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(16);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(19);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(4);
    *b = b.wrapping_add(*a);
}
fn final_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(14));
    *a ^= *c;
    *a = a.wrapping_sub(c.rotate_left(11));
    *b ^= *a;
    *b = b.wrapping_sub(a.rotate_left(25));
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(16));
    *a ^= *c;
    *a = a.wrapping_sub(c.rotate_left(4));
    *b ^= *a;
    *b = b.wrapping_sub(a.rotate_left(14));
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(24));
}

fn valid_call(call: &str) -> Option<String> {
    let c = call.trim().to_ascii_uppercase();
    if c.len() < 3
        || c.len() > 12
        || !c
            .bytes()
            .all(|x| x.is_ascii_uppercase() || x.is_ascii_digit() || x == b'/')
    {
        return None;
    }
    Some(c)
}

fn pair_hash(calls: &str) -> u16 {
    let mut bytes = [b' '; 37];
    let raw = calls.as_bytes();
    bytes[..raw.len()].copy_from_slice(raw);
    let init = 0xdead_beefu32.wrapping_add(37).wrapping_add(146);
    let (mut a, mut b, mut c) = (init, init, init);
    for block in bytes[..36].as_chunks::<12>().0 {
        a = a.wrapping_add(u32::from_le_bytes(block[0..4].try_into().unwrap()));
        b = b.wrapping_add(u32::from_le_bytes(block[4..8].try_into().unwrap()));
        c = c.wrapping_add(u32::from_le_bytes(block[8..12].try_into().unwrap()));
        mix(&mut a, &mut b, &mut c);
    }
    a = a.wrapping_add(bytes[36] as u32);
    final_mix(&mut a, &mut b, &mut c);
    (c & 0x0fff) as u16
}

fn shorthand_bits(hash12: u16, report: usize) -> [u8; 40] {
    let value = (hash12 << 4) | (report as u16);
    let mut message = [0u8; 16];
    for (i, bit) in message.iter_mut().enumerate() {
        *bit = ((value >> i) & 1) as u8; // Fortran message(i)=ishft(ig,1-i)
    }
    let mut reordered = [0u8; 32];
    for (i, row) in GENERATOR.iter().enumerate() {
        let mut parity = 0u8;
        for (j, bit) in message.iter().enumerate() {
            parity ^= *bit & (((row >> (15 - j)) & 1) as u8);
        }
        reordered[COLORDER[i]] = parity;
    }
    for j in 0..16 {
        reordered[COLORDER[16 + j]] = message[j];
    }
    let mut bits = [0u8; 40];
    bits[..8].copy_from_slice(&SYNC);
    bits[8..].copy_from_slice(&reordered);
    bits
}

// WSJT-X bpdecode40.f90's (32,16) parity-check matrix in transmitted
// codeword order. Kept separate from mfsk-core's LDPC(128,90) codec.
const CHECKS: [&[usize]; 16] = [
    &[1, 7, 12, 18, 21, 27, 31],
    &[2, 8, 13, 18, 24, 28],
    &[2, 9, 14, 19, 23, 29],
    &[3, 8, 12, 20, 23, 30],
    &[4, 8, 15, 19, 22, 31],
    &[1, 6, 13, 19, 25, 30],
    &[5, 9, 16, 21, 25, 26],
    &[3, 10, 11, 18, 26, 32],
    &[6, 9, 15, 20, 27, 32],
    &[5, 10, 13, 22, 27],
    &[4, 7, 14, 20, 26, 28],
    &[4, 10, 12, 17, 25, 29],
    &[1, 11, 16, 22, 28, 29],
    &[2, 11, 15, 17, 21, 30],
    &[3, 6, 14, 16, 24, 31],
    &[5, 7, 17, 23, 24, 32],
];

/// Standalone log-domain BP core following WSJT-X bpdecode40.f90.
///
/// Positive LLR means bit=1. Returns the 16-bit MSK40 hash+report payload
/// after parity checks pass; an accepted codeword alone is NOT an accepted
/// on-air decode: sync, signal quality, hash legitimacy and frame validation
/// belong to the surrounding receiver.
#[allow(dead_code)]
fn decode_soft_codeword(llr: &[f32; 32]) -> Option<u16> {
    if llr.iter().any(|x| !x.is_finite() || x.abs() > 1e6) {
        return None;
    }
    let mut check_to_bit = [[0f32; 7]; 16];
    let mut posterior = [0f32; 32];
    for _ in 0..=5 {
        posterior.copy_from_slice(llr);
        for (check, bits) in CHECKS.iter().enumerate() {
            for (edge, pos) in bits.iter().enumerate() {
                posterior[*pos - 1] += check_to_bit[check][edge];
            }
        }
        let hard = posterior.map(|x| if x > 0.0 { 1u8 } else { 0 });
        let valid = CHECKS
            .iter()
            .all(|bits| bits.iter().fold(0u8, |xor, pos| xor ^ hard[*pos - 1]) == 0);
        if valid {
            let mut payload = 0u16;
            for bit in 0..16 {
                payload |= (hard[COLORDER[16 + bit]] as u16) << bit;
            }
            return Some(payload);
        }
        // Fortran bpdecode40: toc=posterior-tov; tanhtoc=tanh(-toc/2)
        // then tov=2*atanh(-product_other(tanhtoc)). Clip near ±1 for
        // numerical stability without turning impossible parity into NaN.
        let mut next = [[0f32; 7]; 16];
        for (check, bits) in CHECKS.iter().enumerate() {
            for edge in 0..bits.len() {
                let mut product = 1f32;
                for other in 0..bits.len() {
                    if other == edge {
                        continue;
                    }
                    let pos = bits[other] - 1;
                    let extrinsic = posterior[pos] - check_to_bit[check][other];
                    product *= (-0.5 * extrinsic).tanh();
                }
                next[check][edge] = 2.0 * (-product).clamp(-0.999_999, 0.999_999).atanh();
            }
        }
        check_to_bit = next;
    }
    None
}

fn tones(hash12: u16, report: usize) -> [u8; 40] {
    let bits = shorthand_bits(hash12, report);
    let sign = |i: usize| 2 * bits[i] as i32 - 1;
    let mut out = [0u8; 40];
    for i in 0..20 {
        let t1 = (sign(2 * i + 1) * sign(2 * i) + 1) / 2;
        let t2 = -(sign(2 * i + 1) * sign((2 * i + 2) % 40) - 1) / 2;
        out[2 * i] = (1 - t1) as u8;
        out[2 * i + 1] = (1 - t2) as u8;
    }
    out
}

fn phases(tones: &[u8; 40]) -> [f64; FRAME_SAMPLES] {
    let mut phi = 0.0f64;
    let mut out = [0.0f64; FRAME_SAMPLES];
    for (k, phase) in out.iter_mut().enumerate() {
        phi += core::f64::consts::TAU * (1000.0 + 1000.0 * f64::from(tones[k / NSPS])) / FS as f64;
        if phi > core::f64::consts::TAU {
            phi -= core::f64::consts::TAU;
        }
        *phase = phi;
    }
    out
}

pub fn synthesize(call1: &str, call2: &str, report: &str, period_s: u32) -> Option<Vec<f32>> {
    if ![5, 10, 15, 30].contains(&period_s) {
        return None;
    }
    let a = valid_call(call1)?;
    let b = valid_call(call2)?;
    let report_idx = REPORTS
        .iter()
        .position(|r| r.eq_ignore_ascii_case(report.trim()))?;
    let hash = pair_hash(&format!("{a} {b}"));
    let frame = tones(hash, report_idx);
    let length = (period_s as usize) * FS - 6000;
    let mut phi = 0.0f64;
    let mut amplitude = 0.7f64;
    let mut audio = Vec::with_capacity(length);
    for k in 0..length {
        let tone = frame[(k / NSPS) % FRAME_TONES];
        phi += core::f64::consts::TAU * (1000.0 + 1000.0 * f64::from(tone)) / FS as f64;
        if phi > core::f64::consts::TAU {
            phi -= core::f64::consts::TAU;
        }
        if k >= length.saturating_sub(204) {
            amplitude *= 0.98f64.powi(4);
        }
        audio.push((amplitude * phi.sin()) as f32);
    }
    Some(audio)
}

/// Only the *known-call* offline receiver. Normalise correlations against
/// every one of the 16 valid report codewords and do NOT guess unknown hashes.
/// Caller must not expose this as general on-air RX acceptance.
pub fn decode_known_calls(pcm: &[f32], call1: &str, call2: &str) -> Option<String> {
    let a = valid_call(call1)?;
    let b = valid_call(call2)?;
    if pcm.len() < FRAME_SAMPLES * 3 || pcm.iter().any(|s| !s.is_finite()) {
        return None;
    }
    let hash = pair_hash(&format!("{a} {b}"));
    let refs: Vec<([f64; FRAME_SAMPLES], [f64; FRAME_SAMPLES], f64)> = (0..16)
        .map(|r| {
            let p = phases(&tones(hash, r));
            let mut sin = [0.0f64; FRAME_SAMPLES];
            let mut cos = [0.0f64; FRAME_SAMPLES];
            let mut norm = 0.0f64;
            for k in 0..FRAME_SAMPLES {
                sin[k] = p[k].sin();
                cos[k] = p[k].cos();
                norm += sin[k] * sin[k];
            }
            (sin, cos, norm)
        })
        .collect();
    // Track a maximum over offsets PER report. Adjacent offsets of the
    // same report are not independent candidates for runner-up rejection.
    let mut scores = [0.0f64; 16];
    // Sliding by one sample allows arbitrary receive-phase alignment.
    // A bounded initial 1-second acquisition window keeps CPU use finite.
    let search_limit = pcm.len().saturating_sub(FRAME_SAMPLES).min(FS);
    for offset in 0..=search_limit {
        if offset % 3 != 0 {
            continue;
        }
        let slice = &pcm[offset..offset + FRAME_SAMPLES];
        let energy = slice
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>();
        if energy < 0.001 {
            continue;
        }
        for (idx, (sin, cos, norm)) in refs.iter().enumerate() {
            let mut si = 0.0f64;
            let mut co = 0.0f64;
            for k in 0..FRAME_SAMPLES {
                si += f64::from(slice[k]) * sin[k];
                co += f64::from(slice[k]) * cos[k];
            }
            let score = (si * si + co * co).sqrt() / (energy * norm).sqrt();
            if score > scores[idx] {
                scores[idx] = score;
            }
        }
    }
    let mut ranks: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
    ranks.sort_by(|a, b| b.1.total_cmp(&a.1));
    let winner = ranks[0];
    if winner.1 < 0.90 || winner.1 - ranks[1].1 < 0.05 {
        return None;
    }
    Some(format!("<{a} {b}> {}", REPORTS[winner.0]))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parity_and_sync_bits_are_deterministic() {
        let bits = shorthand_bits(0xabc, 15);
        assert_eq!(&bits[..8], &SYNC);
        for i in 0..16 {
            assert_eq!(bits[8 + COLORDER[16 + i]], ((0xabcfu16 >> i) & 1) as u8);
        }
    }
    #[test]
    fn independent_wsjt_msk40_reference_vector() {
        // WSJT-X genmsk40.f90 + encode_msk40.f90, Jenkins nhash init 146
        // on the Fortran 37-character blank-padded "K1ABC W9XYZ" field.
        assert_eq!(pair_hash("K1ABC W9XYZ"), 0x0fa2);
        let bits = shorthand_bits(0x0fa2, 14);
        assert_eq!(
            bits.iter()
                .map(|b| char::from(b'0' + *b))
                .collect::<String>(),
            "1011000100001100010010100111010001001111"
        );
        let symbols = tones(0x0fa2, 14);
        assert_eq!(
            symbols
                .iter()
                .map(|b| char::from(b'0' + *b))
                .collect::<String>(),
            "1000011001000001100010111100100110000101"
        );
    }
    #[test]
    fn wsjt_ldpc32_soft_core_matches_reference_codeword() {
        let expected = 0xfa2eu16;
        let bits = shorthand_bits(0x0fa2, 14);
        let soft =
            core::array::from_fn::<f32, 32, _>(|k| if bits[k + 8] != 0 { 7.0 } else { -7.0 });
        assert_eq!(decode_soft_codeword(&soft), Some(expected));
        // A hard-sliced bit flip with weak confidence should be repaired
        // only when all 16 parity checks have converged.
        let mut noisy = soft;
        noisy[8] = if bits[16] == 0 { 1.0 } else { -1.0 };
        assert_eq!(decode_soft_codeword(&noisy), Some(expected));
        let bad = [f32::NAN; 32];
        assert_eq!(decode_soft_codeword(&bad), None);
    }

    #[test]
    fn known_call_shorthand_round_trip_and_noise_rejection() {
        let tx = synthesize("K1ABC", "W9XYZ", "RRR", 5).unwrap();
        assert_eq!(tx.len(), 54_000);
        assert_eq!(
            decode_known_calls(&tx, "K1ABC", "W9XYZ").as_deref(),
            Some("<K1ABC W9XYZ> RRR")
        );
        assert_eq!(decode_known_calls(&vec![0.0; 720], "K1ABC", "W9XYZ"), None);
        assert_eq!(decode_known_calls(&tx, "K1ABC", "K2ABC"), None);
    }
}

/// An MSK40 shorthand decode that passed an actual complex analytic-signal
/// synchronization search, soft-symbol demodulation, and LDPC parity check.
#[cfg(any(feature = "fft-rustfft", feature = "fft-extern"))]
#[derive(Clone, Debug)]
pub struct ShortPing {
    pub message: String,
    pub frequency_hz: f32,
    pub t_sec: f32,
    pub sync_quality: f32,
}

/// Reverse MSK144's sync word, per WSJT-X msk40sync.f90:28-53. The first
/// Q-rail bit is delayed by half a symbol, while I-rail bits start at 0.
#[cfg(any(feature = "fft-rustfft", feature = "fft-extern"))]
fn sync_template() -> [Complex32; 42] {
    let mut t = [Complex32::new(0.0, 0.0); 42];
    let b = |i: usize| if SYNC[i] == 0 { -1.0 } else { 1.0 };
    let pp = crate::engine::dsp::msk::half_sine_pulse();
    for j in 0..6 {
        t[j].im = pp[j + 6] * b(0);
    }
    for k in 0..3 {
        for j in 0..12 {
            t[6 + 12 * k + j].im = pp[j] * b(2 + 2 * k);
        }
    }
    for j in 0..12 {
        t[j].re = pp[j] * b(1);
    }
    for j in 0..12 {
        t[12 + j].re = pp[j] * b(3);
    }
    for j in 0..12 {
        t[12 * 2 + j].re = pp[j] * b(5);
    }
    for j in 0..6 {
        t[36 + j].re = pp[j] * b(7);
    }
    t
}

/// Stage-0 WSJT-X msk40spd.f90 two-tone squared-signal burst detector.
/// Scan 240 samples every 60 (5 ms) at 12 kHz and compare FFT power
/// near 2*(fc-500) and 2*(fc+500). This non-coherent stage only supplies
/// timing/CFO hypotheses; no message is accepted without sync + BP checks.
#[cfg(any(feature = "fft-rustfft", feature = "fft-extern"))]
#[derive(Clone, Copy, Debug)]
pub struct TwoToneCandidate {
    pub sample: usize,
    pub frequency_error_hz: f32,
    pub strength: f32,
}

#[cfg(any(feature = "fft-rustfft", feature = "fft-extern"))]
pub fn detect_two_tone_candidates(
    analytic: &[Complex32],
    fc: f32,
    ntol: f32,
) -> Vec<TwoToneCandidate> {
    let mut out = Vec::new();
    if analytic.len() < 720
        || !fc.is_finite()
        || !ntol.is_finite()
        || !(800.0..=2200.0).contains(&fc)
        || !(0.0..=250.0).contains(&ntol)
    {
        return out;
    }
    const NFFT: usize = 240;
    const STEP: usize = 60;
    let nsteps = ((analytic.len() - NFFT) / STEP + 1).min(150);
    let df = FS as f32 / NFFT as f32;
    let nominal_low = 2.0 * (fc - 500.0);
    let nominal_high = 2.0 * (fc + 500.0);
    let mut planner = crate::engine::fft::default_planner();
    let fwd = planner.plan_forward(NFFT);
    let mut metrics = vec![0.0f32; nsteps];
    let mut tonescore = vec![0.0f32; nsteps];
    let mut freqerr = vec![f32::NAN; nsteps];
    let mut window = [0f32; 12];
    for (k, w) in window.iter_mut().enumerate() {
        *w = (1.0 - (core::f32::consts::PI * k as f32 / 12.0).cos()) / 2.0;
    }
    for step in 0..nsteps {
        let start = step * STEP;
        let mut transformed = Vec::with_capacity(NFFT);
        for k in 0..NFFT {
            let mut z = analytic[start + k] * analytic[start + k];
            if k < 12 {
                z *= window[k];
            }
            if k >= NFFT - 12 {
                z *= window[NFFT - 1 - k];
            }
            transformed.push(z);
        }
        fwd.process(&mut transformed);
        let powers: Vec<f32> = transformed.iter().map(|v| v.norm_sqr()).collect();
        let peak = |center: f32| -> Option<(usize, f32, f32, f32)> {
            let lo = (((center - 2.0 * ntol) / df).round() as isize).clamp(1, 118) as usize;
            let hi = (((center + 2.0 * ntol) / df).round() as isize).clamp(1, 118) as usize;
            if lo > hi {
                return None;
            }
            let index = (lo..=hi).max_by(|a, b| powers[*a].total_cmp(&powers[*b]))?;
            let amp = powers[index];
            let others = (lo..=hi)
                .filter(|i| *i != index)
                .map(|i| powers[i])
                .sum::<f32>();
            let avg = others / (hi - lo + 1) as f32;
            let num = transformed[index - 1] - transformed[index + 1];
            let den = transformed[index] * 2.0 - transformed[index - 1] - transformed[index + 1];
            let interp = if den.norm_sqr() > 1e-16 {
                -(num / den).re.clamp(-0.5, 0.5)
            } else {
                0.0
            };
            Some((index, amp, amp / (avg + 0.01), (index as f32 + interp) * df))
        };
        let Some(hi) = peak(nominal_high) else {
            continue;
        };
        let Some(lo) = peak(nominal_low) else {
            continue;
        };
        metrics[step] = hi.1.max(lo.1);
        tonescore[step] = hi.2.max(lo.2);
        freqerr[step] = if hi.1 >= lo.1 {
            (hi.3 - nominal_high) / 2.0
        } else {
            (lo.3 - nominal_low) / 2.0
        };
    }
    let mut sorted = metrics.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let background = sorted[(nsteps / 4).min(nsteps - 1)].max(1e-6);
    for item in &mut metrics {
        *item /= background;
    }
    // WSJT-X msk40spd first selects <=5 coarse energy peaks >=3.5,
    // then supplements using tone peak-vs-mean ratios >=12.
    for phase in 0..2 {
        while out.len() < 5 {
            let current = if phase == 0 {
                &mut metrics
            } else {
                &mut tonescore
            };
            let Some(index) = (0..nsteps).max_by(|a, b| current[*a].total_cmp(&current[*b])) else {
                break;
            };
            let score = current[index];
            if score < (if phase == 0 { 3.5 } else { 12.0 }) {
                break;
            }
            current[index] = 0.0;
            if !freqerr[index].is_finite() || freqerr[index].abs() > ntol {
                continue;
            }
            if out
                .iter()
                .any(|c: &TwoToneCandidate| c.sample.abs_diff(index * STEP) < STEP)
            {
                continue;
            }
            out.push(TwoToneCandidate {
                sample: index * STEP + 1,
                frequency_error_hz: freqerr[index],
                strength: score,
            });
        }
    }
    out
}

#[cfg(any(feature = "fft-rustfft", feature = "fft-extern"))]
fn demodulate_frame(c: &[Complex32; 240], template: &[Complex32; 42]) -> Option<[f32; 40]> {
    let pp = crate::engine::dsp::msk::half_sine_pulse();
    let phase = c[..42]
        .iter()
        .zip(template)
        .fold(Complex32::new(0.0, 0.0), |acc, (a, b)| acc + *a * b.conj());
    if phase.norm_sqr() < 1e-8 {
        return None;
    }
    let derotate = phase.conj() / phase.norm();
    let mut r = [Complex32::new(0.0, 0.0); 240];
    for k in 0..240 {
        r[k] = c[k] * derotate;
    }
    let mut soft = [0.0f32; 40];
    for k in 0..6 {
        soft[0] += r[k].im * pp[k + 6] + r[234 + k].im * pp[k];
    }
    for k in 0..12 {
        soft[1] += r[k].re * pp[k];
    }
    for i in 2..=20 {
        let start_q = (i - 1) * 12 - 6;
        let start_i = (i - 1) * 12;
        for k in 0..12 {
            soft[2 * i - 2] += r[start_q + k].im * pp[k];
            soft[2 * i - 1] += r[start_i + k].re * pp[k];
        }
    }
    Some(soft)
}

/// Decode actual analytic PCM, rather than ranking synthetic known-message
/// templates. Bounded frequency/timing search with WSJT-X MSK40 reversed
/// sync, half-sine OQPSK matched filters, BP(32,16), and post-BP quality
/// gates. Caller provides a pair of known call signs to avoid presenting
/// arbitrary 12-bit hashes as proven station identities.
///
/// This standalone implementation is intentionally not represented as the
/// fully-calibrated msk40spd.f90 2-tone spectral acquisition; sensitivity
/// and false-positive rates require independent WSJT-X golden validation.
#[cfg(any(feature = "fft-rustfft", feature = "fft-extern"))]
pub fn decode_analytic(
    analytic: &[Complex32],
    fc: f32,
    ntol: f32,
    call1: &str,
    call2: &str,
) -> Option<ShortPing> {
    let a = valid_call(call1)?;
    let b = valid_call(call2)?;
    if analytic.len() < 720
        || !fc.is_finite()
        || !(800.0..=2200.0).contains(&fc)
        || !ntol.is_finite()
        || !(0.0..=250.0).contains(&ntol)
        || analytic
            .iter()
            .any(|c| !c.re.is_finite() || !c.im.is_finite())
    {
        return None;
    }
    let expected = pair_hash(&format!("{a} {b}"));
    let two_tone = detect_two_tone_candidates(analytic, fc, ntol);
    let sync = sync_template();
    let norm: f32 = sync.iter().map(|c| c.norm_sqr()).sum();
    let mut winners: Vec<(f32, usize, f32)> = Vec::with_capacity(10);
    let delta_step = if ntol <= 30.0 { 3.0 } else { 6.0 };
    let nsteps = (ntol / delta_step).ceil() as i32;
    let twopi = 2.0 * core::f32::consts::PI;
    for fi in -nsteps..=nsteps {
        let freq = fc + (fi as f32) * delta_step;
        if freq < fc - ntol - 0.001 || freq > fc + ntol + 0.001 {
            continue;
        }
        let dphi = -twopi * freq / FS as f32;
        let step = Complex32::new(dphi.cos(), dphi.sin());
        let mut rot = Complex32::new(1.0, 0.0);
        let mut base = Vec::with_capacity(analytic.len());
        for &v in analytic {
            base.push(v * rot);
            rot *= step;
            // Avoid gradual numerical drift in very long sessions.
            if base.len() % 1024 == 0 {
                rot /= rot.norm().max(1e-20);
            }
        }
        // Coarse 3-sample timing search; shortlisted peaks get fully
        // resolved with BP and the independent codeword quality gate.
        for offset in (0..=base.len() - 240).step_by(3) {
            if !two_tone.is_empty()
                && !two_tone.iter().any(|candidate| {
                    offset.abs_diff(candidate.sample) <= 3 * FRAME_SAMPLES
                        && (freq - (fc + candidate.frequency_error_hz)).abs() <= 35.0
                })
            {
                continue;
            }
            let mut dot = Complex32::new(0.0, 0.0);
            let mut energy = 0.0f32;
            for k in 0..42 {
                let z = base[offset + k];
                dot += z * sync[k].conj();
                energy += z.norm_sqr();
            }
            let score = dot.norm() / (energy * norm).sqrt().max(1e-9);
            if score < 0.65 {
                continue;
            }
            if winners.len() < 10 {
                winners.push((score, offset, freq));
            } else if let Some((index, _)) = winners
                .iter()
                .enumerate()
                .min_by(|x, y| x.1.0.total_cmp(&y.1.0))
                && score > winners[index].0
            {
                winners[index] = (score, offset, freq);
            }
        }
        // Fine BP only for global best trials, not for every candidate.
    }
    winners.sort_by(|x, y| y.0.total_cmp(&x.0));
    for (score, offset, freq) in winners {
        if score < 0.74 {
            continue;
        }
        let mut frame = [Complex32::new(0.0, 0.0); 240];
        let step = Complex32::new(
            (-twopi * freq / FS as f32).cos(),
            (-twopi * freq / FS as f32).sin(),
        );
        let phase = -twopi * freq * (offset as f32) / FS as f32;
        let mut rot = Complex32::new(phase.cos(), phase.sin());
        for k in 0..240 {
            frame[k] = analytic[offset + k] * rot;
            rot *= step;
        }
        let soft = match demodulate_frame(&frame, &sync) {
            Some(x) => x,
            None => continue,
        };
        let badsync = (0..8)
            .filter(|i| (soft[*i] >= 0.0) != (SYNC[*i] != 0))
            .count();
        if badsync > 3 {
            continue;
        }
        let mean = soft.iter().sum::<f32>() / 40.0;
        let sigma = (soft.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 40.0).sqrt();
        if !sigma.is_finite() || sigma < 1e-6 {
            continue;
        }
        let llr = core::array::from_fn::<f32, 32, _>(|k| 2.0 * soft[k + 8] / (sigma * 0.75 * 0.75));
        let Some(payload) = decode_soft_codeword(&llr) else {
            continue;
        };
        let hash = payload >> 4;
        let report = (payload & 15) as usize;
        if hash != expected || report < 7 {
            continue;
        }
        let reencoded = shorthand_bits(hash, report);
        let mut hard_errors = 0usize;
        let mut mismatches = 0.0f32;
        for k in 0..32 {
            if (soft[k + 8] >= 0.0) != (reencoded[k + 8] != 0) {
                hard_errors += 1;
                mismatches += (soft[k + 8] / sigma).abs();
            }
        }
        if hard_errors > 4 || mismatches >= 0.65 {
            continue;
        }
        return Some(ShortPing {
            message: format!("<{a} {b}> {}", REPORTS[report]),
            frequency_hz: freq,
            t_sec: (offset + 120) as f32 / FS as f32,
            sync_quality: score,
        });
    }
    None
}

#[cfg(all(test, feature = "fft-rustfft"))]
mod acquisition_tests {
    use super::*;
    use crate::engine::dsp::analytic_signal;
    #[test]
    fn decodes_actual_complex_short_burst_not_just_waveform_match() {
        let audio = synthesize("K1ABC", "W9XYZ", "RRR", 5).unwrap();
        let mut received = vec![0.0f32; 7168];
        received[920..920 + 240 * 5].copy_from_slice(&audio[..240 * 5]);
        let analytic = analytic_signal(&received);
        let candidates = detect_two_tone_candidates(&analytic, 1500.0, 12.0);
        assert!(
            !candidates.is_empty(),
            "WSJT-X two-tone stage must detect a clean MSK40 ping"
        );
        assert!(
            candidates.iter().any(|c| c.sample.abs_diff(920) < 240),
            "candidate timing must be near the real burst"
        );
        let got = decode_analytic(&analytic, 1500.0, 12.0, "K1ABC", "W9XYZ");
        assert_eq!(
            got.as_ref().map(|x| x.message.as_str()),
            Some("<K1ABC W9XYZ> RRR")
        );
        assert!(decode_analytic(&analytic, 1500.0, 12.0, "K1ABC", "K2ABC").is_none());
    }
}
