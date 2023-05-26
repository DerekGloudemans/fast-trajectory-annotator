# fast-trajectory-annotator

## TODOs
- [X] Get i24-rcs to publish as a usable repository correctly
- [X] Switch to GPU hardware decoder for frame loading
- [X] Deal with loading data in middle of buffer
- [X] Shift from flat homography to i24-rcs
- [X] Ensure `add()`, `dimension()`, `shift()`, and `copy_paste()` work
- [X] Implement fast object initializer - we assume all vehicles are Nissan Rogues which makes things pretty easy
- [X] Design data structure for holding object positions (frame indexed)
- [X] Design data structure for holding object completeness (red if there is no sink camera, green otherwise)
- [X] Implement auto-box detection (verify that it still works)
- [X] Add save() and quit() and make sure reload works
- [X] Add timestamp file parser and display time in `plot()`
- [X] Add GPS box data
- [X] Associate objects with GPS objects
- [X] Automatically advance camera and frame after selecting a box based on next GPS location
- [X] Add some method for making sure we don't miss any GPS objects currently in the pipeline
- [X] Make it easy to add 2 annotations per object per frame - added the `hop` function
- [ ] Add last-travel-time list per camera pair and add smart_advance for non GPS-paired vehicles
