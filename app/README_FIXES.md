
v1.3.0 hard filters:
- Reject coins whose first market history is older than 2025-01-01.
- Reject coins whose ATH is on listing / first bars of history.
- Reject windows where crown starts earlier than 15 daily bars after listing.
- Reject crowns longer than 60 daily bars.
- Reject windows where ATH is too late inside the crown.
- Reject windows where ATH is too late in the full window.
- Keep explicit reference windows for RIVER and SIREN passable.
Reference windows:
- river: 2025-09-22 -> 2025-12-30
- siren-2: 2025-03-20 -> 2026-02-05
