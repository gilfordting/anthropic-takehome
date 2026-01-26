# todo

- [ ] analyze reg usage for different stages -- what is max in-flight?
- [ ] think about good ways to use register file
- [ ] vbroadcast on the fly to unlock more vreg space?
  - [ ] figure out what to do for this
- [ ] rewrite to use general engine
- [ ] figure out how to do vmerge with the new offset type system
- [ ] no dynamic register offloading, just make some specific ops (in hash?) 8-wide + vmerge
- [ ] how to handle an offset vreg as the dest for something
- [ ] rewrite const creation
  - [ ] add something to provide info about size of freelists after all constants are done
- [ ] dead code elim -- while terminals exist that are not store, remove them
- [ ] all constants should be exported
- [ ] have a report of which constants survived dce; this can be done by just printing out all the leaves at the start
