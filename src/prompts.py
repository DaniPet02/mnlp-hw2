baseline = """
Compile a report containing the results of the qualitative analysis and the aggregate score for Archaic to Modern Italian Traslation.

Adhere to the structure and explain the organization of the report.
evaluation_scale = {
    1: {
        "title": "Completely Unacceptable",
        "description": "Translation bears no resemblance to the original meaning. "
                       "Output is gibberish, nonsensical, or entirely irrelevant."
    },
    2: {
        "title": "Severe Errors",
        "description": "Translation contains critical semantic and/or syntactic errors, "
                       "significant omissions, or unwarranted additions that distort the core message. "
                       "The output is unnatural and clearly not human-like."
    },
    3: {
        "title": "Partially Incorrect / Lackluster",
        "description": "Translation conveys a portion of the original meaning but is marred by "
                       "noticeable errors (e.g., typos, minor semantic inaccuracies, awkward phrasing). "
                       "While understandable, it lacks polish and accuracy."
    },
    4: {
        "title": "Good Translation",
        "description": "Translation is largely accurate and faithful to the original meaning. "
                       "It is fluent, comprehensible, and semantically sound. "
                       "Minor stylistic deviations from the original may be present, but overall quality is high."
    },
    5: {
        "title": "Perfect / Near-Native Translation",
        "description": "Translation is accurate, fluent, complete, and coherent, "
                       "perfectly capturing the meaning, nuance, and style of the original text. "
                       "It reads as if originally written in the target language."
    }
}


"""