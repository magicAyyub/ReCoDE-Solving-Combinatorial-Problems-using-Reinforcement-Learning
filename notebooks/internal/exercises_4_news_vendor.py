# ───────────────────────── Economic‑parameter sampling ──────────────────────── #
economic_parameter_cases = [
    (
        (0,),  # args passed to economic_parameters(...)
        (63.69616873214543,
         17.184380041594505,
          0.20486761968097345,
          0.16527635528529094,
        162.6540478400545),
        "seed 0 -> deterministic (p, c, h, k, μ)"
    ),
    (
        (42,),
        (77.39560485559633,
         33.967262302690486,
          4.292989599556912,
          6.973680290593639,
         18.835469577529906),
        "seed 42 -> different deterministic draw"
    ),
]

first_step_reward_cases = [
    (
        (0, 0.0),        # (seed, action)
        -30.410849372493534,
        "seed 0, action 0 -> reward matches analytic calc"
    ),
    (
        (42, 0.0),
        -160.3946466836537,
        "seed 42, action 0 -> reward matches analytic calc"
    ),
]