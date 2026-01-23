- Iteration 1: single vectorization of valu operations, etc.
  - 31809
- Iteration 2: overlap alu and load for the part that has to be sequentialized
  - 28225
- Iteration 3: FMA for idx calc
  - 27713
- Iteration 4: FMA for hash
  - 26181
- Iteration 5: Switch order of outer loops to amortize load/store
  - 22213
- Iteration 6: Pack store of current batch and load of next batch into one instruction
  - 22152
- Iteration 7: Simplify 3 of the hash functions further; some more parallel packing for hash building. now either 1 or 2 instructions
  - 12937
- Iteration 8: Super easy index wraparound; more compact index update
  - 10857
- Iteration 9: Caching + using nodes 0, 1, 2
  - 9820
- Iteration 10: more caching, but more intelligent
  - 9083
- Iteration 11: two batches at once
  - 4557
- Iteration 12: symbolic execution engine and physical register file partitioning, 32 independent instruction streams, packed const loading, heuristic-based VLIW scheduler
  - 1469

TODO:

- Then, pack more VLIW stuff together

Target: 1487
