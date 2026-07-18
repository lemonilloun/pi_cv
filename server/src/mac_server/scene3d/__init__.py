"""Scene3D: offline room reconstruction from Pi scene sessions.

Pipeline (each step reads the previous step's artifacts under
<session>/derived/):

    depth   — Depth Anything V2 metric per keyframe (MPS)
    poses   — pycolmap sequential SfM + metric scale alignment
    tsdf    — Open3D TSDF fusion -> room mesh + 2D floor plan
    objects — masks + depth + poses -> 3D objects, DINOv2 association
    graph   — CLIP open-vocab labels + near/on scene graph

Driven by pipeline.SceneJobRunner (web tab) or scripts/run_scene_pipeline.sh.
"""
