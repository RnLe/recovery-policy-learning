// Recovery Policy Learning / Grounded Recovery, technical report.
// Compile after `gr analyze`, `gr audit`, and `gr export-typst`:
//   typst compile report/main.typ build/recovery-policy-learning-report.pdf
// Every empirical number is imported from report/generated/, which is written
// by the analysis pipeline from the validated result bundle.

#import "generated/metadata.typ": *

#set document(title: "Recovery Policy Learning / Grounded Recovery")
#set page(margin: (x: 2.4cm, y: 2.6cm), numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set block(above: 1.6em, below: 0.9em)
#show link: underline

// A cross-reference to a chapter shows its name. "Section 21" tells a reader
// nothing they can act on; the chapter's own title does.
#show ref: it => {
  let target = it.element
  if target != none and target.func() == heading {
    link(target.location(), emph(target.body))
  } else {
    it
  }
}

#align(center)[
  #text(size: 17pt, weight: "bold")[Recovery Policy Learning / Grounded Recovery]

  #v(0.3em)
  #text(size: 11.5pt)[
    Budget-matched corrective imitation learning for recovery \
    in language-conditioned embodied policies in BabyAI/MiniGrid
  ]

  #v(0.5em)
  #text(size: 9pt, fill: rgb("#444444"))[
    Protocol #protocol-version · frozen contract, one receipted test opening
  ]
]

#v(1.2em)

#include "chapters/00_summary.typ"
#include "chapters/01_reading_guide.typ"

#pagebreak()
#outline(depth: 1, indent: auto)

#v(0.8em)
#text(size: 9pt, fill: rgb("#444444"))[
  Chapters 3 to 14 are Part I, the confirmatory study. Chapters 15 to 22 are
  Part II, the foundations track, which is exploratory throughout.
]

// ---------------------------------------------------------------------------
#pagebreak()
#align(center)[
  #v(3.2cm)
  #text(size: 16pt, weight: "bold")[Part I · The Grounded Recovery study]
  #v(0.8em)
  #text(size: 10.5pt)[
    The confirmatory experiment, under the frozen contract, \
    with one receipted test opening.
  ]
]

#include "chapters/study/01_question_scope.typ"
#include "chapters/study/02_background.typ"
#include "chapters/study/03_environment_oracle.typ"
#include "chapters/study/04_design_estimand.typ"
#include "chapters/study/05_data_model.typ"
#include "chapters/study/06_verification.typ"
#include "chapters/study/07_results.typ"
#include "chapters/study/08_failure_analysis.typ"
#include "chapters/study/09_exploratory_extensions.typ"
#include "chapters/study/10_discussion_limitations.typ"
#include "chapters/study/11_outlook.typ"
#include "chapters/study/12_reproduction.typ"

// ---------------------------------------------------------------------------
#pagebreak()
#align(center)[
  #v(3.2cm)
  #text(size: 16pt, weight: "bold")[Part II · Foundations]
  #v(0.8em)
  #text(size: 10.5pt)[
    Every ingredient of the study, built from the ground up and measured: \
    the world, the decision problem, the teacher, the learning paradigm, \
    the network, the failure mode, and the measurement discipline.
  ]
  #v(0.8em)
  #text(size: 9pt, fill: rgb("#8a6d1a"))[
    Exploratory teaching material. All confirmatory evidence lives in Part I.
  ]
]

#include "chapters/foundations/f00_roadmap.typ"
#include "chapters/foundations/f01_world.typ"
#include "chapters/foundations/f02_decision.typ"
#include "chapters/foundations/f03_oracle.typ"
#include "chapters/foundations/f04_learning.typ"
#include "chapters/foundations/f05_architecture.typ"
#include "chapters/foundations/f06_shift.typ"
#include "chapters/foundations/f07_measurement.typ"
