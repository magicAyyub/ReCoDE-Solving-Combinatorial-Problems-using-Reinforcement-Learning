import ipywidgets as widgets
from IPython.display import display, HTML, Markdown

def _format_case(case):
    """Normalise a case into (inp, expected, desc)."""
    if len(case) == 2:
        inp, exp = case
        desc = ""
    else:
        inp, exp, desc = case
    return inp, exp, desc

def run_tests(func, cases, *, stop_on_first: bool=False):
    """ Runs a set of test cases (expected outputs) on a function.

    Args:
        func (callable): the function to test
        cases (sequence of input, expected, description | None): sequence of expected input-output cases, possibly named
        stop_on_first (bool):  break at first failure if True.

    Returns:
        List of (passed: bool, message: str)
    """
    results = []
    for i, raw in enumerate(cases, 1):
        inp, exp, desc = _format_case(raw)
        label = f"Test {i} " + (f"({desc})" if desc else "") + " —"
        try:
            # Flexible argument passing
            if isinstance(inp, tuple):
                out = func(*inp)
            elif isinstance(inp, dict):
                out = func(**inp)
            else:
                out = func(inp)
            
            assert out == exp, f"got {out!r}, expected {exp!r}"
            results.append((True, f"✅ {label} passed"))
        except AssertionError as err:
            results.append((False, f"❌ {label} failed: {err}"))
            if stop_on_first:
                break
        except Exception as e:
            results.append((False,
                            f"❌ {label} raised {type(e).__name__}: {e}"))
            if stop_on_first:
                break
    return results

def make_tester(func, cases, stop_on_first: bool = False, *, label: str="Run Tests"):
    """ Return a widget (Button + Output box) that runs the tests when clicked.
    
    Args:
        func (callable): the function to test
        cases  : sequence of (input, expected, [description])
        stop_on_first (bool): break at first failure if True.
        label (str): of the button text to run tests

    Returns:
        widget VBox: run tests button in an output box
    """
    out = widgets.Output()
    button = widgets.Button(description=label,
                         button_style="success",
                         tooltip="Click to check your code",
                         icon="check")

    def _on_click(_):
        out.clear_output()
        with out:
            res = run_tests(func, cases, stop_on_first=stop_on_first)
            passed = all(ok for ok, _ in res)
            if passed:
                display(HTML(
                    "<p style='color:green; font-weight:bold; font-size:1.1em'>"
                    "🎉 All tests passed — great job!</p>"))
            else:
                first_fail = next(msg for ok, msg in res if not ok)
                display(HTML(f"<p style='color:red; font-weight:bold'>{first_fail}</p>"))

            # Detailed per‑case summary
            bullets = "\n".join(f"- {msg}" for _, msg in res)
            display(Markdown(bullets))

    button.on_click(_on_click)
    centered_button = widgets.HBox([button], layout=widgets.Layout(justify_content='center'))

    return widgets.VBox([centered_button, out])