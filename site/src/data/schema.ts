// Shapes of the pipeline-generated JSON the site consumes. These mirror the
// files actually produced by `gr publish-result` and `grf stage-site`; the
// invariants in validate.ts are what make them trustworthy to render.

export type SiteStatus = {
  schema_version: string;
  phase: "protocol" | "pilot" | "results";
  protocol_version: string;
  protocol_hash: string | null;
  code_hash?: string;
  code_commit: string | null;
  generated_at: string;
  result_release: boolean;
  canonical_report: string;
};

export type MethodId = "bc_base" | "extra_demo" | "recovery";
export type SliceId = "clean" | "matched" | "unseen";

export type Outcome = {
  method: MethodId;
  slice: SliceId;
  assigned_episodes: number;
  successful_episodes: number;
  success_rate: number;
  intervention_delivered: number | null;
};

export type Replicate = {
  bundle_id: string;
  outcomes: Outcome[];
  primary_paired_difference: number;
};

export type ExperimentSummary = {
  experiment_id: string;
  generated_at: string;
  protocol: {
    version: string;
    hash: string;
    code_commit: string | null;
    primary_contrast: string;
    primary_endpoint: string;
    sesoi_absolute_success: number;
  };
  methods: Array<{ id: MethodId; label: string }>;
  budget: {
    additional_revealed_targets: { extra_demo: number; recovery: number };
    optimizer_updates_matched: boolean;
    target_exposures_matched: boolean;
    replay_rules_matched: boolean;
  };
  eligibility: {
    candidate_scenarios: number;
    eligible_scenarios: number;
    retained_fraction: number;
    manifest_hash: string;
  };
  replicates: Replicate[];
  primary_summary: {
    analysis_status: "confirmatory" | "exploratory_pilot";
    claim_state: "support" | "rule_out" | "adverse" | "inconclusive";
    mean_paired_difference: number;
    interval: { method: string; level: number; lower: number; upper: number };
    pipeline_replicates: number;
    precision: {
      desired_half_width: number;
      achieved_half_width: number;
      target_met: boolean;
    };
    sensitivity: {
      method: string;
      level: number;
      lower: number;
      upper: number;
      mean?: number;
      replicates?: number;
    };
  };
};

export type MediaItem = {
  id: string;
  href: string;
  poster: string;
  sha256: string;
  empirical: boolean;
  selection_rule: string;
  outcome?: string;
  mission?: string;
  steps?: number;
  scenario_ordinal?: number;
  corruption_time?: number | null;
  trace?: string;
};

export type MediaManifest = {
  schema_version: string;
  fps: number;
  items: MediaItem[];
  trajectories: Array<{ id: string; href: string; sha256: string }>;
};

export type SpectrumEntry = {
  success_rate: number;
  mean_steps: number;
  episodes: number;
};

export type BcSummary = {
  open_loop_accuracy_mean: number;
  unseen_success_mean: number;
  train_success_mean: number;
  unseen_success_per_seed: number[];
};

export type JourneyData = {
  schema_version: string;
  evidence_label: string;
  environment: {
    env_id: string;
    max_steps: number;
    action_names: string[];
    observation_shape: number[];
  };
  lab01: {
    census_seeds: number;
    unique_missions: number;
    grid_shape: string;
    mission_colors: Record<string, number>;
    mission_kinds: Record<string, number>;
  };
  lab02: {
    episodes: number;
    total_states: number;
    observation_classes: number;
    aliased_classes: number;
    cross_world_classes: number;
    conflicting_classes: number;
    memoryless_error_floor: number;
  };
  lab03: {
    spectrum: Record<string, SpectrumEntry>;
    sync: { pairs: number; success_rates: Record<string, number> };
  };
  lab04: {
    qlearning: { states: number; best_steps: number; episodes: number };
    dataset: { episodes: number; labels: number };
    memoryless: BcSummary;
    recurrent: BcSummary;
  };
  lab05: {
    parameters: number;
    walkthrough: Array<[string, number, string]>;
    variants: Array<{
      name: string;
      parameters: number;
      unseen_success_mean: number;
      unseen_success_per_seed: number[];
    }>;
  };
  lab06: {
    design: Record<string, unknown> & { budget_labels: number };
    success_matrix: Record<string, Record<SliceId, number>>;
    per_rep_unseen_delta: number[];
    delivered_rate: number;
    exposure: Record<string, number>;
    sweep: {
      times: number[];
      delivered_success: number[];
      clean_success: number;
    };
  };
  lab07: {
    simulation: Record<string, number>;
    pairing: Record<string, number>;
    reanalysis: Record<string, unknown>;
    unmatched: { budget_labels: number; unseen_success: number } | null;
    hash_chain: Record<string, unknown>;
    contract_hash_demo: Record<string, string>;
  };
  media: MediaManifest;
};
