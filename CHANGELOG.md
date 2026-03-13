# CHANGELOG

## [Release] v0.2.0 — Squelch Híbrido (Auto/Manual)

### O que foi modificado

- ✨ `crates/novasdr-core/src/protocol.rs` — Adicionado campo `level: Option<f32>` ao `ClientCommand::Squelch` com `#[serde(default)]` para retrocompatibilidade
- ✨ `crates/novasdr-server/src/state.rs` — Adicionado `squelch_level: Option<f32>` ao `AudioParams`
- ✨ `crates/novasdr-server/src/ws/audio.rs` — Implementado squelch manual com histerese de 10 frames, cálculo de `pwr_db` espelhando matemática do S-Meter, roteamento automático via `Option<f32>`
- ✅ `crates/novasdr-server/src/ws/audio.rs` — 4 novos testes unitários: `squelch_manual_opens_when_power_above_threshold`, `squelch_manual_closes_with_hysteresis`, `squelch_manual_resets_hysteresis_on_signal_return`, `squelch_auto_ignores_pwr_db`
- ♻️ `crates/novasdr-server/src/ws/audio.rs` — Refatorados 4 testes existentes para nova assinatura do `SquelchState::update`
- ✨ `frontend/src/components/audio/types.ts` — Adicionados campos `squelchAuto: boolean` e `squelchLevel: number` ao `AudioUiSettings`
- ✨ `frontend/src/components/audio/useAudioClient.ts` — Payload squelch envia `level` com compensação de offset do S-Meter
- ✨ `frontend/src/components/receiver/panels/AudioPanel.tsx` — Sub-seção condicional com Switch "Auto" e Slider de nível (-140 a 0 dB)
- ✨ `frontend/src/App.tsx` — Defaults `squelchAuto: true`, `squelchLevel: -80`
