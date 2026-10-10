"""Gradio movie-picker demo; no public sharing enabled by default."""

import argparse

import gradio as gr

from src.recommend import Recommender


def build_demo(directory: str) -> gr.Interface:
    """Use explicit (title, ID) choices so duplicate titles do not corrupt mapping."""
    model = Recommender(directory)

    def recommend(liked: list[int]) -> list[list[str]]:
        return [[r["title"], r["explanation"]] for r in model.recommend(liked=liked)]

    return gr.Interface(
        fn=recommend,
        api_name="recommend",
        analytics_enabled=False,
        inputs=gr.Dropdown(
            choices=[(title, i) for i, title in enumerate(model.titles)],
            multiselect=True,
            label="Movies you like",
        ),
        outputs=gr.Dataframe(headers=["Recommendation", "Explanation"]),
        title="Synthetic recommendation demo"
        if model.dataset_kind == "synthetic"
        else "MovieLens recommendations",
        description="Research demo. Explanations describe genre/year similarity; preferences may not be representative.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts/demo")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_demo(args.artifacts).launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
