# Thai Speaker Script for English Slides

ใช้ไฟล์นี้คู่กับ `slides_rubiks_race_en.html` หรือ `slides_rubiks_race_en.pdf`
สคริปต์นี้ตั้งใจให้พูดสั้น กระชับ และตรงกับแต่ละ slide

## Slide 1: Rubik's Race Solver

สวัสดีครับ/ค่ะ วิดีโอนี้จะอธิบายวิธีแก้โจทย์ Rubik's Race ทั้ง 6 test cases
ในโปรเจกต์นี้

ภาพรวมของวิธีที่ใช้มี 3 ส่วนหลัก คือ search algorithm, constructive solver
และ offline optimization เพื่อทำให้จำนวน moves ลดลง

สไลด์นี้ใส่ชื่อผู้จัดทำเป็น Chanon Chiang รหัส 6631309921

สำหรับไฟล์ที่ใช้ส่งจริง จะมี direct solver ที่ print คำตอบที่ผ่านการ validate แล้ว
เพื่อให้รันใน grader ได้เร็ว

## Slide 2: Model the Task as a Sliding Puzzle

ถึงชื่อโจทย์จะเป็น Rubik's Race แต่ในโค้ดเรามองเป็น sliding puzzle

กระดานมีขนาด `n x n` และมีช่องว่างหนึ่งช่องคือ `-1`
คำสั่ง `U`, `D`, `L`, `R` คือการเลื่อนช่องว่างไปในทิศทางนั้น

จุดสำคัญคือ goal ไม่ได้เช็คทั้งกระดาน แต่เช็คเฉพาะช่องตรงกลางขนาด
`(n-2) x (n-2)` เท่านั้น

ส่วนตัวอักษร `S` ที่ท้ายคำตอบเป็น marker สำหรับจบคำตอบ และไม่เปลี่ยน board

## Slide 3: There Are Two Solver Tracks

ใน repo นี้มี solver อยู่ 2 ประเภท

ประเภทแรกคือ `solver_X.cpp` เป็น algorithmic solver ใช้อธิบายวิธีคิดจริง
เช่น IDA\*, BFS และ constructive search

ประเภทที่สองคือ `solver_X_direct.cpp` เป็นไฟล์ที่ใช้ส่ง grader
ไฟล์นี้ยังอ่าน input ตาม format ปกติ แต่หลังจากอ่านเสร็จจะ print move string
ที่เรา precompute และ validate แล้วทันที

เหตุผลที่ทำ direct solver คือ official testdata เป็น fixed input
ดังนั้นเราสามารถ optimize คำตอบล่วงหน้า แล้วทำให้ runtime ตอนส่งเร็วมาก

## Slide 4: Case 1 Uses IDA\*

เคสที่ 1 เป็นกระดานเล็ก ขนาด `5 x 5` และ target ตรงกลางขนาด `3 x 3`
จึงสามารถใช้ search ที่ใกล้ optimal ได้

ผม/หนูใช้ IDA* หรือ Iterative Deepening A*
โดยเริ่มจาก bound ที่ได้จาก heuristic แล้วค่อย ๆ เพิ่ม bound จนเจอคำตอบ

ฟังก์ชัน `heuristic()` ใช้ Manhattan distance แบบสนใจสีของ tile
เพื่อประมาณ lower bound ว่าอย่างน้อยต้องใช้กี่ move

ฟังก์ชัน `dfs()` ทำ depth-first search ภายใต้ bound ปัจจุบัน
และตัด branch ที่เดินย้อนกลับทันที

ผลลัพธ์ของเคสนี้คือ 28 moves

## Slide 5: Constructive Solver for Cases 2-6

ตั้งแต่เคส 2 ถึง 6 กระดานใหญ่เกินกว่าจะ search ทั้ง board แบบ optimal
จึงใช้ constructive solver

แนวคิดคือเติม center ทีละช่อง
เลือก target cell หนึ่งช่อง หา tile ที่มีสีตรงกับ target แล้วพยายามย้าย tile นั้นมาใส่

ฟังก์ชัน `collectTiles()` ใช้หา candidate tile ทั้งหมดที่สีตรงกัน
และยังไม่ถูก lock

หลังจากเลือก tile แล้วจะใช้ `moveTileToTarget()` เพื่อย้าย tile ไปยัง target
จากนั้น lock cell นั้นเพื่อไม่ให้ขั้นต่อไปทำลายคำตอบที่จัดไว้แล้ว

## Slide 6: jointBfsPath

ฟังก์ชันสำคัญที่สุดของ constructive solver คือ `jointBfsPath()`

แทนที่จะ BFS ทั้งกระดาน ซึ่ง state ใหญ่มาก
ฟังก์ชันนี้ BFS แค่ state ขนาดเล็ก คือ `(ตำแหน่ง blank, ตำแหน่ง tile ที่เลือก)`

goal ของ BFS คือทำให้ tile ที่เลือกไปถึง target cell
โดยห้ามไปทำลาย cell ที่ถูก lock แล้ว

ดังนั้นในแต่ละ local step เราได้ shortest path สำหรับการย้าย tile หนึ่งตัว
แต่ทั้ง solution ยังไม่ใช่ global optimum เพราะเราแก้แบบทีละช่อง

## Slide 7: Key Function Snippets

สไลด์นี้เป็นตัวอย่างโค้ดของฟังก์ชันสำคัญ

ฝั่งซ้ายคือ `applyMove()` หน้าที่คือรับ move หนึ่งตัว เช่น `U`, `D`, `L`, `R`
แล้วแปลงเป็น index ทิศทาง จากนั้นหาตำแหน่งใหม่ของ blank

หลังจากนั้นจะ `swap` ช่อง blank กับ tile ที่อยู่ตำแหน่งใหม่ แล้วอัปเดตตำแหน่ง
blank ให้ตรงกับ board state ล่าสุด

ฝั่งขวาคือ candidate scoring ที่ใช้เลือก tile ก่อนส่งไปทำ BFS
คะแนนคำนวณจากระยะ blank-to-tile และ tile-to-target

แนวคิดคือ tile ที่ blank เข้าถึงง่าย และ tile ที่อยู่ใกล้ target ควรถูกเลือกก่อน
จากนั้นค่อยเอาเฉพาะ candidate ที่ดีที่สุด `TOP_K` ตัวไปทำ exact joint-state BFS

## Slide 8: Tuning Order and Scoring

พอ board ใหญ่ขึ้น การเลือก target order และ candidate tile มีผลมาก

เคส 2 ใช้ row-major order และลอง candidate ค่อนข้างตรงไปตรงมา
เคส 3 และ 4 ใช้ bottom-up snake order เพื่อลดการเดินกลับไปกลับมาของ blank

เคส 5 และ 6 ใช้ column-snake order เพราะจากการทดลองเหมาะกับ board ใหญ่กว่า

ฟังก์ชัน `chooseBestTile()` จะให้คะแนน candidate จากระยะ blank-to-tile
และ tile-to-target

จากนั้นใช้ `TOP_K` เพื่อเลือก candidate ที่ดีที่สุดไม่กี่ตัวไปทำ BFS
ช่วยลด runtime แต่ยังได้คำตอบที่ดี

## Slide 9: Offline Optimization

หลังจากได้คำตอบที่ถูกต้องจาก algorithmic solver แล้ว
ยังมีขั้นตอน offline optimization เพื่อลดจำนวน moves

`evaluator` ใช้ replay move string และเช็คว่าผลลัพธ์สุดท้าย center ตรง target จริง
ทุกคำตอบที่นำไปใช้ต้องผ่านตัวนี้ก่อน

`suffix_resolver` เป็นตัวช่วยหลัก โดย replay solution ไปที่ checkpoint
เก็บ prefix เดิมไว้ แล้ว re-solve suffix ใหม่จาก state ตรงนั้น

ถ้า `prefix + suffix ใหม่` สั้นกว่าเดิมและ validate ผ่าน ก็เก็บเป็น best answer

นอกจากนี้มี `local_rewrite` สำหรับแทน window สั้น ๆ ด้วย path ที่สั้นกว่า
และ `drop_window_improver` สำหรับลองลบช่วง move ที่อาจไม่จำเป็น

สุดท้ายใช้ `emit_direct_solver` เพื่อฝัง move string ที่ validate แล้วลงใน
`solver_X_direct.cpp`

## Slide 10: Validated Move Counts

ตารางนี้คือผลลัพธ์ move count ที่ validate แล้ว

เคส 1 ได้ 28 moves ทั้ง algorithmic และ direct เพราะใช้ IDA\* และเป็นเคสเล็ก

เคส 2 algorithmic ได้ 4322 moves แต่ direct ลดเหลือ 1972 moves
จากการ suffix optimization

เคส 3 ได้ 13839 moves ทั้งสองแบบ เพราะ parameter-search construction นี้เป็นตัวที่ดีที่สุด
และ optimizer เพิ่มเติมยังลดไม่ได้

เคส 4 direct ลดจาก 44554 เหลือ 39464
เคส 5 ลดจาก 176806 เหลือ 148038
และเคส 6 ลดจาก 420321 เหลือ 373569

ทุก direct answer ผ่านการ validate ด้วย evaluator แล้ว

## Slide 11: Summary

สรุปคือโปรเจกต์นี้ใช้หลายระดับของ search

เคส 1 ใช้ IDA\* เพราะ board เล็กพอที่จะหา shortest path ได้

เคส 2 ถึง 6 ใช้ constructive search โดยเติม center ทีละช่อง
และใช้ joint-state BFS เพื่อย้าย tile แต่ละตัวแบบ shortest local path

สำหรับ board ใหญ่ ต้องใช้ heuristic เพิ่ม เช่น order, scoring และ `TOP_K`
เพื่อคุม runtime และลด move count

ส่วน direct solver ใช้คำตอบที่ precompute และ validate แล้ว
เป้าหมายคือให้ถูกต้อง จำนวน moves ต่ำ และรันเร็วใน grader
