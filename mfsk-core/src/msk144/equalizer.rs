//! WSJT-X msk144signalquality.f90 / analytic.f90-inspired session adaptive
//! MSK144 RX equalizer. Never auto-trains on unverified/unknown calls.
//!
//! The trained five-term polynomial is fitted to the *complex cross-spectrum*
//! between a CRC/LDPC-confirmed demodulated frame and its independently
//! reconstructed 77-bit transmit waveform. The receive filter applies only
//! quadratic and higher phase terms (WSJT-X deliberately discards gain,
//! constant phase, and linear delay terms).
//! Training is station-scoped; a target change or reset invalidates it.

use alloc::string::String;
use num_complex::Complex32;
#[cfg(not(feature = "std"))]
use num_traits::Float;

use crate::engine::fft::default_planner;
use crate::msk144::tx::{codeword, synth_codeword_frame};

const N: usize = 864;
const POINTS: usize = 145; // +/-1000 Hz around the 1500 Hz MSK carrier
const HALF: isize = 72;

#[derive(Clone, Debug)]
pub struct AdaptiveEqualizerStatus {
    pub enabled: bool,
    pub trained: bool,
    pub training_frames: u32,
    pub residual_rms_rad: f64,
    pub coefficients: [f64; 5],
}

#[derive(Clone, Debug)]
pub struct AdaptiveEqualizer {
    enabled: bool,
    target: String,
    trained: bool,
    frames: u32,
    last_time_s: Option<f64>,
    cross: [Complex32; POINTS],
    weight_sum: f64,
    coeff: [f64; 5],
    residual: f64,
}

impl Default for AdaptiveEqualizer {
    fn default() -> Self {
        Self {
            enabled: false,
            target: String::new(),
            trained: false,
            frames: 0,
            last_time_s: None,
            cross: [Complex32::new(0.0, 0.0); POINTS],
            weight_sum: 0.0,
            coeff: [0.0; 5],
            residual: f64::INFINITY,
        }
    }
}

impl AdaptiveEqualizer {
    /// Changing target or enabling state resets training, without leaking
    /// coefficients learned from another station or radio chain.
    pub fn configure(&mut self, enabled: bool, target: &str) {
        let target = target.trim().to_ascii_uppercase();
        if !enabled
            || target.is_empty()
            || !target
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'/')
        {
            *self = Self::default();
            return;
        }
        if self.enabled && self.target == target {
            return;
        }
        *self = Self::default();
        self.enabled = true;
        self.target = target;
    }

    pub fn status(&self) -> AdaptiveEqualizerStatus {
        AdaptiveEqualizerStatus {
            enabled: self.enabled,
            trained: self.trained,
            training_frames: self.frames,
            residual_rms_rad: self.residual,
            coefficients: self.coeff,
        }
    }

    /// Only expose coefficients once all five distinct high-quality frames
    /// agree on a low-residual curve.
    pub fn coefficients(&self) -> Option<&[f64; 5]> {
        self.trained.then_some(&self.coeff)
    }

    /// A single successful frame may be used once. At least 5 independent,
    /// strong frames associated with the configured DX call are required.
    pub fn observe(
        &mut self,
        rx: &[Complex32],
        info77: &[u8; 77],
        message: &str,
        hard_errors: u32,
        snr_db: f32,
        time_s: f64,
    ) -> bool {
        if !self.enabled
            || self.trained
            || rx.len() != N
            || self.target.is_empty()
            || !message.to_ascii_uppercase().contains(&self.target)
            || !time_s.is_finite()
            || !snr_db.is_finite()
            || snr_db <= 5.0
            || hard_errors >= 7
        {
            return false;
        }
        if let Some(last) = self.last_time_s {
            if (time_s - last).abs() <= 0.072 {
                return false;
            }
        }
        // The ideal model is generated from the verified 77 information bits,
        // never from arbitrary user text or a weak spectral peak.
        let model = synth_codeword_frame(&codeword(info77));
        let mut model_fft = model.to_vec();
        let mut rx_fft = rx.to_vec();
        let mut planner = default_planner();
        let fwd = planner.plan_forward(N);
        fwd.process(&mut model_fft);
        fwd.process(&mut rx_fft);
        let weight = 10f32.powf(snr_db / 20.0);
        for k in 0..POINTS {
            let signed = k as isize - HALF;
            let bin = if signed < 0 {
                (N as isize + signed) as usize
            } else {
                signed as usize
            };
            let product = model_fft[bin] * rx_fft[bin].conj();
            self.cross[k] += product * (weight / 1000.0);
        }
        self.weight_sum += weight as f64;
        self.frames += 1;
        self.last_time_s = Some(time_s);

        if self.frames < 5 {
            return false;
        }
        match self.fit_phase() {
            Some((coeff, rms))
                if rms < 0.5 && coeff[2..].iter().all(|v| v.is_finite() && v.abs() < 2.5) =>
            {
                self.coeff = coeff;
                self.residual = rms;
                self.trained = true;
                true
            }
            _ => {
                // WSJT-X retrains rather than publishing bad coefficients.
                if self.frames >= 10 {
                    self.frames = 0;
                    self.last_time_s = None;
                    self.cross = [Complex32::new(0.0, 0.0); POINTS];
                    self.weight_sum = 0.0;
                }
                false
            }
        }
    }

    fn fit_phase(&self) -> Option<([f64; 5], f64)> {
        let mut a = [[0f64; 6]; 5];
        let mut phases = [0f64; POINTS];
        let mut weights = [0f64; POINTS];
        let mut previous = None::<f64>;
        let df = 12_000.0 / N as f64 / 1000.0;
        let mut sum_w = 0.0;
        for i in 0..POINTS {
            let cross = self.cross[i];
            let w = cross.norm() as f64;
            if !w.is_finite() || w < 1e-8 {
                continue;
            }
            let mut phase = (cross.im as f64).atan2(cross.re as f64);
            if let Some(prev) = previous {
                while phase - prev > core::f64::consts::PI {
                    phase -= core::f64::consts::TAU;
                }
                while phase - prev < -core::f64::consts::PI {
                    phase += core::f64::consts::TAU;
                }
            }
            previous = Some(phase);
            phases[i] = phase;
            weights[i] = w;
            sum_w += w;
            let x = (i as isize - HALF) as f64 * df;
            let basis = [1.0, x, x * x, x * x * x, x * x * x * x];
            for row in 0..5 {
                for col in 0..5 {
                    a[row][col] += w * basis[row] * basis[col];
                }
                a[row][5] += w * basis[row] * phase;
            }
        }
        if sum_w < 1e-6 {
            return None;
        }
        // Small 5x5 weighted LS via pivoted Gaussian elimination.
        for k in 0..5 {
            let pivot = (k..5).max_by(|x, y| a[*x][k].abs().total_cmp(&a[*y][k].abs()))?;
            a.swap(k, pivot);
            let d = a[k][k];
            if d.abs() < 1e-12 || !d.is_finite() {
                return None;
            }
            for j in k..6 {
                a[k][j] /= d;
            }
            for row in 0..5 {
                if row == k {
                    continue;
                }
                let factor = a[row][k];
                for j in k..6 {
                    a[row][j] -= factor * a[k][j];
                }
            }
        }
        let coeff = core::array::from_fn::<f64, 5, _>(|k| a[k][5]);
        let mut residual = 0.0;
        for i in 0..POINTS {
            if weights[i] <= 0.0 {
                continue;
            }
            let x = (i as isize - HALF) as f64 * df;
            let fit = coeff[0] + x * (coeff[1] + x * (coeff[2] + x * (coeff[3] + x * coeff[4])));
            residual += weights[i] * (fit - phases[i]).powi(2);
        }
        Some((coeff, (residual / sum_w).sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::msg::wsjt77::pack77;
    #[test]
    fn learns_a_deliberately_distorted_phase_curve_from_five_frames() {
        let info = pack77("K1ABC", "K2ABC", "-10").unwrap();
        let model = synth_codeword_frame(&codeword(&info));
        let mut spectrum = model.to_vec();
        let mut plan = default_planner();
        let fwd = plan.plan_forward(N);
        let inv = plan.plan_inverse(N);
        fwd.process(&mut spectrum);
        // The channel rotates RX by the negative of the 0.45*x^2+0.12*x^3
        // phase response. The learned correction must reverse it.
        for (k, value) in spectrum.iter_mut().enumerate() {
            let signed = if k <= N / 2 {
                k as isize
            } else {
                k as isize - N as isize
            };
            let x = signed as f32 * (12_000.0 / N as f32) / 1000.0;
            let phase = -(0.45 * x * x + 0.12 * x * x * x);
            *value *= Complex32::new(phase.cos(), phase.sin());
        }
        inv.process(&mut spectrum);
        for v in &mut spectrum {
            *v /= N as f32;
        }
        let mut eq = AdaptiveEqualizer::default();
        eq.configure(true, "K2ABC");
        for k in 0..5 {
            eq.observe(&spectrum, &info, "K1ABC K2ABC -10", 0, 12.0, k as f64);
        }
        let status = eq.status();
        assert!(
            status.trained,
            "high-confidence phase-distorted frames must train"
        );
        assert!(
            (status.coefficients[2] - 0.45).abs() < 0.08,
            "quadratic phase correction {}",
            status.coefficients[2]
        );
        assert!(
            (status.coefficients[3] - 0.12).abs() < 0.08,
            "cubic phase correction {}",
            status.coefficients[3]
        );
        assert!(status.residual_rms_rad < 0.1);
    }

    #[test]
    fn requires_five_independent_verified_frames_and_target() {
        let info = pack77("K1ABC", "K2ABC", "-10").unwrap();
        let model = synth_codeword_frame(&codeword(&info));
        let mut eq = AdaptiveEqualizer::default();
        eq.configure(true, "K2ABC");
        assert!(!eq.observe(&model, &info, "K1ABC K3ABC -10", 0, 12.0, 0.1));
        assert!(!eq.observe(&model, &info, "K1ABC K2ABC -10", 8, 12.0, 0.2));
        for k in 0..4 {
            assert!(!eq.observe(&model, &info, "K1ABC K2ABC -10", 0, 12.0, k as f64));
        }
        let _ = eq.observe(&model, &info, "K1ABC K2ABC -10", 0, 12.0, 5.0);
        assert!(
            eq.status().trained,
            "clean model should train a flat equalizer"
        );
        assert!(
            eq.coefficients().unwrap()[2..]
                .iter()
                .all(|x| x.abs() < 0.05)
        );
        eq.configure(true, "K3ABC");
        assert!(!eq.status().trained);
        assert_eq!(eq.status().training_frames, 0);
        eq.configure(false, "K2ABC");
        assert!(!eq.status().enabled);
    }
}
