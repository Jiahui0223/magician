"""EXP5 Task 3 / E1.2 -- manifest of the S_B2 fixed states: the frozen ablation s0_states rule applied to the Task 3 B2
closed loops (source label V2M = the planner service records of B2), same times. CPU. Output W3/e12/abroot_B2/manifests."""
import e12_paths as P

A = P.set_stratum("B2")
A.SOURCES = ("V2M",)
import s0_states  # noqa: E402

if __name__ == "__main__":
    s0_states.main()
