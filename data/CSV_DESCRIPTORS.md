# CSV descriptors for `data/`

This folder contains leader speed profiles and standard drive-cycle traces used
as inputs to the car-following simulations.

## Drive-cycle files

Applies to:

- `Artemis_mw.csv`
- `Artemis_rural.csv`
- `Artemis_urban.csv`
- `Ford_focus.csv`
- `FTP.csv`
- `HWFET.csv`
- `WLTP_class_2_DCycle.csv`

Each file contains one speed trace.

| Column | Unit | Description |
|---|---:|---|
| `Time in s` | s | Time stamp from the start of the drive cycle. |
| `Speed in kmph` | km/h | Leader vehicle speed at the corresponding time. |

Rows are ordered by increasing time. Each row is one sampled instant of the
drive cycle. The simulation code converts speeds as needed when building the
leader speed profile.
