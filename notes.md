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

# analysis/observations

- everything starts at 0 index, so we don't actually need to load indices
- everything completes in one-cycle, no stalling
- we're severely bottlenecked on load slots because of the NUMA pattern, where we load from `values`
  - how much does caching help us?
- height is 10, this means number of nodes is 2047
  - this won't fit in scratch space, but half of it can! (1024)
- some basic bottleneck analysis:

## control flow

- tree depth proceeds in a very predictable fashion:
  - 0 -> 2 -> 6 -> 14 -> 30 -> 62 -> 126 -> 254 -> 510 -> 1022 -> 2046 -> 0
  - on round 11 is when we circle back, so round is effectively mod tree_height+1
    - end of round 10 is when we set indices to 0. this is tree_height
  - definitely feasible to avoid loading for rounds 0, 1, 2 (11, 12, 14 also)
- wraparound check only needs to happen on round 10
  - this isn't even a check, we just set them all to 0

## arithmetic

- several of the hash functions (0, 2, 4) are (x + const) + (x \*2^n); this can be turned into a single multiply-add like const + x\* (1 + 2^n)
- index update can be turned into 2*idx + 2 - (val % 2)
- hash can probably be pipelined across multiple batches/vectors at once (6 valu slots)

## loading

- if we switch the order of the loops, we only need to load `val` once; amortized across all rounds
  - this is 64 vloads in total, 32 instructions
- we don't need to load indices, since they all start at 0
- completely random data access eats up load slots, because they can't be vectorized
  - worst case: this is 1 load per item per round, so 16*256 = 4096 loads, 2048 instructions. this cannot do
  - notice that the top portion of the tree will be reused more than the bottom half, because it's less spread out
- how can we load the tree into scratch (vectorized) and then distribute appropriately?
  - seems reasonable to broadcast to treeval0,1,2...7 to do some stuff
  - this will get more complicated the further down you go

## VLIW scheduling

- probably need some sort of scheduler where we can see ops to be scheduled?
  - would be cool to make a scoreboard

## bottleneck analysis

- 16 rounds and batch size of 256

### load

- load initial val for each item, vectorized: 256/8 = 64 loads/2 = 32 cycles; this is unavoidable
- how about random access? naive, serialized gather with no caching needs 1 per round per item, so 256*16 = 4096 / 2 = 2048 cycles
- if we cache N of the top layers, they're used twice (up to a certain point)
- so 256*(16-2N)/2 = 2048 - 256N
- 32+2048 - 256N < 1400? N > 2.65. this makes sense
  - N = 3 means 1312 is possible if we have perfect packing
  - N = 4 means we can get to 1056, and this is only two vloads
  - N = 5 means 800, but seems a lot hairier? 4 vloads
- how are we going to make use of cache?
  - round 0: one possibility
  - round 1: two possibilities, 1 vselect
  - round 2: 4, 2 vselects
  - round 3: 8, 3 vselects
  - round 4: 8, 4 vselects
- conclusion: num cycles when caching N of the top layers is 2080 - 256N

### alu/valu

- alu throughput is 12 + 6*8 = 60 ops/cycle
- multiply_add counts as one op
- 3 of the hash stages are 3 ops, 3 of them can be expressed as just 1 op; in total this is 12 ops
- 16 rounds, 256 items: 49,152 ops required. 819.2 cycles; this is a more fundamental bottleneck?
  - this implies going past N = 5 for caching won't help us, since the bottleneck shifts here

### flow

- selects are used to filter through the cache
- round 0: no filtering,

## scratch space analysis

- 8 scalar global vars
- 3 scalar constants
- 2 address registers
- 3x8 vector constants
- 3x8 vector variables
- 3x8 vector temporaries
- 6x9 vectorized hash constants + scalar counterpart
- (2^N-1)x8 vectorized forest values
- i_base constants: 256/8 = 64 of these
- 203 + 8(2^N-1) = 1536 -->  N = 7.39 as max cache depth? 128 cached items.
- Might have to adjust this after VLIW pipelining, though -- that will probably take linear amt relative to number of active batches

## TODO

- [ ] fix scratch_const; preload and bundle everything

## pseudocode

round 0:
look at val and update idx, treeval
left if even, right if odd
2x+1 if even, 2x+2 if odd

round 0:
val = hash(val ^ tree0)
parity = val % 2
@parallel
  idx = 1 + parity
  treeval = parity \* (v2-v1) + v1

round 1:
@parallel
  val = hash(val ^ treeval)
  idx = 2*idx + 1
@parallel
  parity = val % 2
  tmpidx = idx - 3
  
@parallel
  diff1 = parity \* (v4-v3) + v3
  diff2 = parity \* (v6-v5) + v5
  idx = idx + parity  # this is final
  
@parallel
  ddiff = diff2 - diff1
  tmpidx = tmpidx >> 1
treeval = tmpidx \* ddiff + diff1

round 2+:
@parallel
  val = hash(val ^ treeval)
  idx = 2*idx + 1

parity = val % 2
idx = idx + parity
treeval = gather(idx)

wraparound round:
val = hash(val ^ treeval)
don't need to set idx or treeval, hardcoded in r0

last round:
val = hash(val, treeval)
store val
load init val for next batch
