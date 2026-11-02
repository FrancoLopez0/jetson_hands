# Power analysis and preliminary battery sizing

> **Evidence level: ONE independent measurement, lasting 86.52 seconds.**
> These are exploratory observations and theoretical battery calculations, not
> repeated-run results, a validated full-system power budget, or guaranteed
> operating times. The CSV and JSON are two exports of the **same recording**.

## 1. Recording and provenance

Raw inputs and the machine-readable calculation are archived in
[`measurements/2026-11-02-single-run/`](../measurements/2026-11-02-single-run/README.md).

| Item | Recorded value |
|---|---:|
| Independent recording sessions | **1** |
| CSV / JSON samples | 87 / 86 |
| CSV start, UTC | 2026-11-02 02:51:36.173 |
| CSV end, UTC | 2026-11-02 02:53:02.693 |
| Elapsed monotonic time | 86.52 s |
| Mean sample interval | 1.00605 s |
| Shortest / longest interval | 1.005 / 1.008 s |

The JSON was exported about one second before the CSV. Its 86 samples match the
CSV in every compared numeric field, including all six cores' clocks and usage.
There are no missing JSON sample IDs in this export. An extra CSV row is not a
second experiment or a data inconsistency. The supplied screenshot ends earlier
than the files, at 02:52:56.657 UTC.

Original exports are preserved byte-for-byte, including the old power wording.
SHA256 hashes are recorded in `analysis.json`; Git attributes disable newline
conversion for these inputs. They contain local source/session metadata but no
access token, video or landmarks.

## 2. Measurement boundary: module, not complete kit

NVIDIA identifies Orin Nano/NX `VDD_IN` as **Total Module Power**. The original
UI/export label "board input" was too broad and has been corrected. Do not treat
the old wording as a whole-developer-kit or battery-terminal measurement.

| Channel | Interpretation |
|---|---|
| `VDD_IN` | Total module power |
| `VDD_CPU_GPU_CV` | Combined CPU/GPU/CV rail, not separate CPU and GPU watts |
| `VDD_SOC` | SoC rail serving the memory subsystem and other engines |

The carrier's power tree determines which peripheral loads are inside or outside
the boundary. Complete carrier, camera, storage, fan, communications and external
DC/DC losses have not been separately measured. Do not assume all are included,
or add them again without checking what is already counted. Do not add the two
subrails to `VDD_IN`: that double-counts part of the input.

## 3. Observations from this one trace

| Statistic | Value |
|---|---:|
| Time-weighted mean `VDD_IN` | **7.653395 W** |
| Arithmetic sample mean | 7.653931 W |
| Lowest / highest sampled power | 7.542 / 7.739 W |
| Descriptive population standard deviation | 0.060027 W |
| Integrated energy | **662.172 J = 0.183937 Wh** |
| GPU clock throughout the recording | 306 MHz |

Integration uses actual monotonic intervals, not an assumed exact one second:

```text
energy_J = sum((power[i-1] + power[i]) / 2 * (time[i] - time[i-1]))
energy_Wh = energy_J / 3600
mean_power_W = energy_J / elapsed_seconds
```

The standard deviation describes samples inside one short recording. It is not
sensor accuracy, uncertainty across independent trials, or a confidence interval.
The sampled maximum is not an electrical transient or startup limit.

### The large drop is GPU utilization, not total CPU utilization

These descriptive windows were selected from the trace. They are not independent
experiments or proven tracker-state labels. Transitional samples outside them
remain in the overall energy calculation.

| UTC window | Samples | CPU mean | GPU usage | GPU clock | VDD_IN | CPU/GPU/CV rail |
|---|---:|---:|---:|---:|---:|---:|
| 02:51:36.173–02:52:09.375 | 34 | 15.87% | 32.47% | 306 MHz | 7.699 W | 1.701 W |
| 02:52:13.399–02:52:33.519 | 21 | 16.25% | 17.00% | 306 MHz | 7.579 W | 1.655 W |
| 02:52:44.585–02:53:02.693 | 19 | 15.76% | 31.21% | 306 MHz | 7.700 W | 1.713 W |

Between the first two windows, GPU utilization decreases about **47.6% relative**,
while CPU mean increases **0.37 percentage points**. Module input decreases only
**0.120 W**, about **1.55%**. The implied runtime gain on the same battery is
about **1.6%**, not a doubling of autonomy.

This is consistent with the user's report of hand acquisition: searching runs
the palm detector repeatedly; successful tracking uses the landmark crop and
skips palm detection. **The exports do not record hand count, `palm_ran`, tracking
state or FPS**, so these files alone cannot establish the causal timing.

Load also migrates between cores. At 02:52:51.626, CPU5 is at 1% and CPU3 is at
75%, yet the online-core mean remains 16.17%. Watching CPU5 alone would suggest
a larger load reduction. CPU clocks and brief DVFS spikes are sampled snapshots,
not direct battery-energy measurements.

Mean CPU/GPU/CV rail power is 1.687 W and SoC rail power is 2.593 W. Approximately
5.967 W of module input is outside the CPU/GPU/CV rail; approximately 3.374 W is
not broken down by the two displayed subrails. Do not assign that residual to a
particular component without the power tree. A large utilization reduction need
not produce a comparable reduction of total input power.

## 4. Battery calculation assumptions

Use watt-hours, not amp-hours alone:

```text
nominal_pack_Wh = nominal_pack_V * pack_Ah
runtime_h = nominal_pack_Wh * converter_efficiency * usable_fraction / load_W
```

All tables use **assumed** 90% external DC/DC efficiency and 80% usable nominal
battery energy: `0.90 * 0.80 = 0.72`. Neither factor was measured; both battery
families use the same factors. Cutoff, aging, temperature, balance and conversion
topology may change them. The 80% is a planning allowance, not a universal spec.

- **7.653 W column:** extrapolation of measured **module-only** load; optimistic
  for the complete system even with the reserve.
- **10 W column:** illustrative complete-load scenario, **NOT a measurement**
  or a validated upper bound for this kit.
- If power is measured before the external converter, its losses are already
  included and must not be applied a second time.

No battery was discharged in this capture. Every runtime is a calculation, not
an endurance test. Nominal Wh approximates energy across the discharge curve;
it is not maximum open-circuit voltage multiplied by Ah.

## 5. Custom 18650 packs

Reference: **3.6 V nominal, 3.5 Ah typical, approximately 12.6 Wh per cell**,
matching published Molicel INR-18650-M35A parameters. The 18650 designation is a
format, not a capacity or chemistry. For otherwise equivalent 3.0 Ah cells,
multiply the energies/runtimes by `3.0 / 3.5 = 0.857`.

`S` means cells in series; `P` means parallel branches. **3S2P** means six cells,
**10.8 V nominal / 7 Ah / 75.6 Wh**. Series adds voltage; parallel adds Ah.
These are representative configurations, not an exhaustive list.

| Configuration | Cells | Nominal V | Nominal Wh | Hours at measured 7.653 W | Hours at assumed 10 W |
|---|---:|---:|---:|---:|---:|
| 2S1P | 2 | 7.2 | 25.2 | 2.4 | 1.8 |
| 3S1P | 3 | 10.8 | 37.8 | 3.6 | 2.7 |
| 4S1P | 4 | 14.4 | 50.4 | 4.7 | 3.6 |
| **3S2P** | **6** | **10.8** | **75.6** | **7.1** | **5.4** |
| **4S2P** | **8** | **14.4** | **100.8** | **9.5** | **7.3** |
| 5S2P | 10 | 18.0 | 126.0 | 11.9 | 9.1 |
| 6S2P | 12 | 21.6 | 151.2 | 14.2 | 10.9 |

Times are decimal hours with the assumed loss/reserve factors included.
Rearranging the same cells does not create energy: 3S2P and 6S1P both use six
reference cells and contain 75.6 Wh nominal. Current, conversion requirements
and cutoff differ and can change practical efficiency.

## 6. Power-tool batteries

For **18 V nominal** packs, `Wh = 18 * Ah`. DEWALT specifies 20V MAX as **18 V
nominal**; its DCB205 5 Ah represents approximately **90 Wh**, not 100 Wh. Makita
BL1850B is also specified as 18 V / 5 Ah. Check each actual pack's nominal Wh and
condition; marketing voltages and brand interfaces are not interchangeable.

| 18 V nominal tool pack | Nominal Wh | Hours at measured 7.653 W | Hours at assumed 10 W |
|---|---:|---:|---:|
| 2 Ah | 36 | 3.4 | 2.6 |
| 3 Ah | 54 | 5.1 | 3.9 |
| 4 Ah | 72 | 6.8 | 5.2 |
| **5 Ah** | **90** | **8.5** | **6.5** |
| 6 Ah | 108 | 10.2 | 7.8 |
| 8 Ah | 144 | 13.5 | 10.4 |

For 90 Wh, ignoring all losses gives about 11.76 h at the measured module load.
The stated 0.72 factor gives **8.47 h**; the hypothetical 10 W complete load gives
**6.48 h**. None is guaranteed operating time for this developer kit.

### Direct comparisons

| Custom pack | Nearby tool pack | Nominal energy difference |
|---|---|---:|
| 3S2P, 75.6 Wh | 18 V / 4 Ah, 72 Wh | Custom has 5% more |
| 4S2P, 100.8 Wh | 18 V / 5 Ah, 90 Wh | Custom has 12% more |
| 5S2P, 126 Wh | 18 V / 6 Ah, 108 Wh | Custom has 16.7% more |

At equal usable Wh and converter efficiency, runtime is comparable. A high
discharge-current rating alone does not extend autonomy. The 90 Wh tool pack
has approximately 19% more energy than the 75.6 Wh custom 3S2P pack.

## 7. Preliminary selection and power path

- **Prototype convenience:** existing genuine 18 V / 5 Ah tool pack with matching
  charger: 8.5 h module-load reference, or 6.5 h at assumed 10 W. Benefits are
  housing, replacement and charging convenience, with extra bulk and adapter/
  protection compatibility to evaluate.
- **Compact custom integration:** 3S2P: 7.1 h / 5.4 h respectively, with freedom
  over layout. Requires suitable BMS, balancing and matched charging.
- **More custom-pack energy:** 4S2P: 9.5 h / 7.3 h respectively, with two more cells.

These are test candidates, not a final electrical design or purchase list.
Do not select a direct connection from nominal voltage alone. Match regulated
output to verified carrier requirements and the original supply. For an
**assumed 19 V regulated output**, lower-voltage packs need boost; an 18 V tool
pack can pass above and below the target during discharge, so a buck-only design
cannot be assumed to extract the planned energy. Evaluate buck-boost over the
complete usable range and include low-voltage cutoff. Do not assume a mechanical
tool adapter reproduces all protections implemented by the tool/pack system.

## 8. Still unmeasured

1. Complete power path with the final camera, storage, fan, communications and
   converter. Battery-side measurement directly includes conversion losses.
2. Repeated longer runs covering search, stationary/moving tracking, stream
   disconnect, startup and representative thermal/environmental conditions.
3. Tracker state, hand count, palm invocations and FPS alongside telemetry, to
   label states rather than infer them from utilization.
4. Transients: the maximum of a 1 Hz, 86.52-second trace does not bound boot,
   radio, storage or GPU current excursions.
5. Real discharge/cutoff with the chosen pack. Evaluate 30 vs 60 FPS or power
   modes with measured Wh and tracking quality; halving FPS need not halve power.

**There is still only one power recording behind this document.** Unit tests
validate data checks and arithmetic, not battery endurance, electrical sensor
accuracy, or coverage of the complete system.

## Sources

- [NVIDIA R36.4.4: Orin Nano/NX channel definitions](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#jetson-orin-nx-series-and-jetson-orin-nano-series)
- [Molicel M35A: 3.6 V nominal, 3.5 Ah typical](https://www.molicel.com/product/inr-18650-m35a/)
- [DEWALT DCB205: 5 Ah and 18 V nominal disclaimer](https://www.dewalt.com/product/dcb205/20v-max-5ah-battery)
- [Makita BL1850B: 18 V / 5 Ah](https://www.makitatools.com/products/details/BL1850B)
