# machine

## core

- scratch space (only one type of memory)
- trace_buf
- pc

## memory

- ALU only operates on register memory
- but more regular mem available

## vliw engines

- generally:
  - reference scratch memory with integer; like a bunch of registers R0, R1, R2, etc. will read from core.scratch
  - have scratch_write and mem_write
- slots:
  - 12 alu
  - 6 vectorized alu
  - 2 load
  - 2 store
  - 1 flow
- ALU: op, dest, a1, a2
  - dest <- a1 op a2
  - operates only in scratch mem
  - ops: +-*, floor/cdiv, booleans, shifts, mod, < and =
- valu
  - scratch only
  - broadcast: dest, src; dest is base reg
  - multiply_add: dest = a*b + c
  - also regular ops
- load
  - all write to scratch
  - load: dest <- addr, addr taken from register and go mem --> scratch
  - load_offset: dest, addr, offset
  - vload: dest, addr; addr
    - for both, addr is base
  - const: immediates; dest, val
- store
  - store: mem <- scratch; addr in reg
  - vstore: base mem addr and scratch mem
- flow
  - select; basically a ternary
  - add_im: add constant
  - vselect: ternary with predication vector
  - halt, pause: turn core on and off
  - trace_write: log info
  - cond_jump: loop
  - cond_jump_rel: jump based on offset
  - jump: always jump
  - indirect, addr stored in register
  -

# problem

- perfect balanced binary tree, values on nodes
- heap structure, and you index into values

## hash

- (a + 0x7ED55D16) + (a << 12) is one stage

## algo

- num nodes is 1 less than pow of 2
- run `h` rounds
- then we have a given number of inputs
- each input will hash against given node
- then go left or right depending on value

## mem layout

- header of 7: # rounds, num tree values, num index/value pairs, tree height, pointer to forest values, pointer to input indices, pointer to input values
- extra room for tree values, input indices&values, vlen*2, 32
- tree vals, input indices, input values

## strategy

- you effectively have one-cycle loads
- there is no penalty for fully unrolling the loop -- no reason to use loop constructs here?
  - besides the select, basically no reason to have control flow
- 6 valu and 12 alu means 60 ops per cycle
  - way less than the 16 load/store per cycle
  - in steady state we'll want to use everything?
- scratch size is 1536, which is 192 vregs
- valu has FMA, which can be used for some hash stages and index update
- switch the loops; this allows for copying to be amortized across rounds
