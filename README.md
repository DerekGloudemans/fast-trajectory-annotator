# fast-trajectory-annotator

## TODOs
- [X] Get i24-rcs to publish as a usable repository correctly
- [ ] Shift from flat homography to i24-rcs
- [X] Switch to GPU hardware decoder for frame loading
- [ ] Deal with loading data in middle of buffer
- [ ] Implement fast object initializer - arrow-key toggle through classes,arrow-key shift object position, array key adjust dimensions , press enter
- [ ] Create array for expected travel times for each lane and each camera pair
- [ ] Implement select_lane() 
- [ ] Implement predict_next_frame_idx()
- [ ] Design data structure for holding object positions (frame indexed)
- [ ] Design data structure for holding object completeness (red if there is no sink camera, green otherwise)
- [ ] Implement auto-box detection (verify that it still works)
- [ ] Add a timer to keep track of how long a single object takes
- [ ] Implement time analysis graph
- [ ] Automatically advance camera and frame after selecting a box
- [ ] Implement queue of n last vehicles annotated and store crops for each (to prevent double labeling).
