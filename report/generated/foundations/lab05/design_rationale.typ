// GENERATED. DO NOT EDIT. (gr_foundations lab05)
// Evidence status: exploratory foundations material, not confirmatory.
#table(
  columns: 2,
  [*component*],
  [*why*],
  [three channel embeddings],
  [the observation is symbolic (lookup indices, Lab 1), not pixels; object, color, and door-state get separate learned vectors instead of one arbitrary integer scale],
  [two 3x3 convolutions],
  [local spatial patterns over the 7x7 view; two valid convolutions reduce 7x7 to 3x3 before a linear projection],
  [mission GRU],
  [turns the instruction into one vector; the ablation checks whether order-aware encoding beats a bag of words at this grammar size],
  [direction embedding],
  [the view direction is part of the observation (Lab 2); an embedding, not a raw integer],
  [previous-action embedding],
  [the policy knows what was just *executed*, the channel through which an external corruption becomes visible (Lab 6)],
  [fusion layer],
  [concatenate all features, mix once, nonlinearity],
  [policy GRU],
  [the memory that Lab 2 proved necessary: aliased observations demand history-dependence],
  [linear head],
  [three logits, one per frozen action],
)
