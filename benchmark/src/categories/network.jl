# Merges the NetworkSpec-based motifs (mps.jl/ctmrg.jl/trg.jl) into one `:network` category,
# tagged by `params.topic` (`:mps`/`:ctmrg`/`:trg`) for `@tagged`-based filtering -- same pattern
# as `:contract` merging synthetic shapes with TCCG. `sizes` is the shared bond-dimension knob
# (`D` for mps, `chi` for ctmrg/trg); each topic interprets it independently.

_network_cases(sizes) = vcat(_mps_cases(sizes), _ctmrg_cases(sizes), _trg_cases(sizes))

register_category!(
    :network, _network_cases;
    sizes = (16, 24, 32, 48, 64, 96, 100, 128, 256, 300, 512)
)
