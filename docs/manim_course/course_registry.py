"""Discover isolated chapter scripts without changing the approved review."""
from importlib import import_module
from content import CHAPTER_00, CHAPTER_01
from course_content import CHAPTERS as FOUNDATIONS, CODE as FOUNDATION_CODE, TITLES, STEMS


def load_course(require_complete=False):
    chapters = {k: v for k, v in FOUNDATIONS.items() if 2 <= k <= 5}
    code = dict(FOUNDATION_CODE)
    for module_name in ("content_06_08", "content_09_12", "content_13_15"):
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
            continue
        overlap = set(chapters) & set(module.CHAPTERS)
        if overlap:
            raise ValueError(f"Duplicate chapter owners: {overlap}")
        chapters.update(module.CHAPTERS)
        for key, source in module.CODE.items():
            if key in code and code[key] != source:
                raise ValueError(f"Conflicting executable example {key}")
            code[key] = source
    if require_complete and set(chapters) != set(range(2, 16)):
        raise ValueError(f"Missing chapters: {set(range(2, 16)) - set(chapters)}")
    return chapters, code


def narration_registry():
    chapters, _ = load_course()
    result = {0: CHAPTER_00, 1: CHAPTER_01}
    result.update({n: {key: (beat.caption, beat.narration, beat.source)
                      for key, beat in chapter.items()} for n, chapter in chapters.items()})
    return result


def code_preludes():
    result = {}
    for name in ("content_06_08", "content_09_12", "content_13_15"):
        try:
            module = import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
            continue
        result.update(getattr(module, "CODE_PRELUDES", {}))
    return result


def extra_inventory():
    strings, formulas = [], []
    for name in ("content_06_08", "content_09_12", "content_13_15"):
        try:
            module = import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
            continue
        strings.extend((name, text) for text in getattr(module, "EXTRA_SCREEN_TEXT", []))
        formulas.extend((name, formula) for formula in getattr(module, "EXTRA_FORMULAS", []))
    return strings, formulas


def stem(number):
    if number == 0:
        return "chapter_00_roadmap"
    if number == 1:
        return "chapter_01_algebra_vectors_products"
    return f"chapter_{number:02}_{STEMS[number]}"
