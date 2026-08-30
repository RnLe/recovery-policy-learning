// The one module that imports gsap. Plugins register here, and everything
// else pulls the configured instance from this file; re-exporting a value
// that callers consume keeps the registration from being tree-shaken.

import { gsap } from "gsap";
import { DrawSVGPlugin } from "gsap/DrawSVGPlugin";
import { MotionPathPlugin } from "gsap/MotionPathPlugin";

gsap.registerPlugin(DrawSVGPlugin, MotionPathPlugin);

export { gsap };
export type Timeline = gsap.core.Timeline;
