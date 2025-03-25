from langgraph.types import interrupt


def ask_user_feedback(context, writer):
    writer("Asking for user feedback...")

    human_feedback = interrupt(
        {
            "task": "Review the generated demo and questions.",
            "generated_idea": context["generated_idea"],
        }
    )

    context["human_feedback"] = human_feedback
    return context
