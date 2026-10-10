# Post-test failure diagnostics

These examples were intentionally selected after scoring to preserve failures alongside successful demonstrations. They are not a new test or representative sample.

V-JEPA uses the development-selected `no_memory` configuration. SAM2 raw masks and calibrated patch predictions are shown separately from explicitly labeled ground-truth visibility references.

- **test_crossing_3100**: Lowest SAM2 visible-localization rate across all test clips; lexicographic scene tie break.
  [Video](sam2_lowest_visible_localization.mp4) · [Diagnostic frame 30](sam2_lowest_visible_localization.png)

- **test_crossing_3104**: Most wrong-car outputs among crossing clips for the development-selected V-JEPA variant; lexicographic scene tie break.
  [Video](vjepa_crossing_wrong_car.mp4) · [Diagnostic frame 58](vjepa_crossing_wrong_car.png)

