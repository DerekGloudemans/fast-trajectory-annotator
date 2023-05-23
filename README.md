# fast-trajectory-annotator

## TODOs
- [X] Get i24-rcs to publish as a usable repository correctly
- [X] Switch to GPU hardware decoder for frame loading
- [X] Deal with loading data in middle of buffer
- [ ] Shift from flat homography to i24-rcs
- [ ] Ensure add(), dimension(), shift(), and copy_paste() work
- [ ] Implement fast object initializer - we assume all vehicles are Nissan Rogues which makes things pretty easy
- [ ] Design data structure for holding object positions (frame indexed)
- [ ] Design data structure for holding object completeness (red if there is no sink camera, green otherwise)
- [ ] Implement auto-box detection (verify that it still works)
- [ ] Add a timer to keep track of how long a single object takes
- [ ] Automatically advance camera and frame after selecting a box
