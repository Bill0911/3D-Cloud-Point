## Algorithm Parameter Report: Room Segmentation Logic (for SOR las/laz file)

### 1. Executive Summary
 - The room segmentation algorithm uses **Reflected Ceiling Plan** approach. Instead of analyzing the floor (which is cluttered with furniture), the team analyzes the top of the apartment.
 - The method relies on 4 key parameters. This document explains what they are, why the matter, and how the team tuned them to achieve the best result as possible.

---

### 2. Parameter Impact Table (Quick Reference)

| **Parameter**      | **If set too LOW…**                                                | **If set too HIGH…**                                   |
|--------------------|--------------------------------------------------------------------|---------------------------------------------------------|
| **Slice Thickness** | Walls disappear; Rooms merge.                                       | Furniture appears; Phantom rooms created.               |
| **Ceiling Offset**  | Ceiling lights appear as noise.                                     | We miss the top of the wall headers.                    |
| **Erosion Radius**  | Doors don't separate (Under-segmentation).                          | Small rooms disappear (Over-segmentation).              |
| **Resolution**      | Processing is slow, Noise increases.                                | Detail is lost, thin walls vanish.                      |

---

### 3. The Slicing Parameters In Details (Finding the Walls)
These settings determine what the computer sees.
#### A. Slice Thickness (The Wafer Slice Technique)
Definition: How deep of a slice the user takes from the ceiling downwards.
- The Sweet Spot: **0.17m - 0.28m** (depending on wall header depth).
- It matters because:
### Slice Thickness Effects

| **If too Thin (< 0.15m)**                     | **If too Thick (> 0.30m)**                                                                                                      |
|----------------------------------|------------------------------------------------------------------------------------------------------------------|
|    The computer misses the "headers" above doors. It thinks the Living Room and Hallway are one giant room because it can't see the wall between them.     | The slice hits the top of tall furniture (Kitchen Cabinets, Fridge). The computer thinks the fridge is a wall, creating "phantom rooms" or black holes in the map |

* The tuning: The team uses Adaptive Slicing computational task that combines a thin layer (to save low-ceiling rooms) with a deep layer (to catch walls).

#### B. Ceiling Offset
Definition: The safety buffer the user skips at the very top before we start slicing.
- The Sweet Spot: **0.05m - 0.06m (5-6 cm)**.
- It matters because:
* Ceilings are never perfectly flat. They have texture, paint, and surface-mounted lights.
* The team skims the top 6cm to make sure clean air and walls are cut through, not light fixture or smoke detectors.

### 4. The Processing Parameter (Defining the Rooms)
These settings determine how the computer thinks.
#### C. Erosion Radius (The "Doorway Separator")
Definition: The algorithm temporarily shrinks the open space of a room to break the connection between areas.
- The Sweet Spot: **0.30m - 0.35m**.
- It matters because: The doorway is usually 0.80m wide. If we shrink the room by 0.35m from both sides (0.70m total), the 'water' connecting the rooms recedes, and the connection breaks.

| **If too Weak (< 0.20m)**                     | **If too Strong (> 0.50m)**                                                                                                      |
|----------------------------------|------------------------------------------------------------------------------------------------------------------|
|    The connection remains. The Living Room floods into the Kitchen.     | The user accidentally scrubs away thin walls or split a narrow hallway into fragmented pieces. |
#### D. Voxel Resolution (The "Pixel Size")
Definition: The level of detail for the digital map.
- The Sweet Spot: **0.05m (5cm)**.
- It matters because: Real interior walls are 10-15cm thick, so 5cm is the ideal block size to represent a wall.

| **If smaller (1cm)**                     | **If larger (10cm)**                                                                                                      |
|----------------------------------|------------------------------------------------------------------------------------------------------------------|
|    The map becomes too noisy, a single missing laser point looks like a hole in the wall     | The pixels are too clunky, narrow walls disappear entirely |

### 5. Conclusion
By calibrating these four parameters, the algorithm will successfully:
- Ignores Furniture: By slicing high (Ceiling logic)
- Detects All Walls: By using Adaptive Slicing (smart merge)
- Separates Rooms: By tuning the Erosion to the width of a standard door.