"""
String transformation rules taken from here:
https://github.com/fastai/fastai/blob/master/fastai/text/transform.py

The IMDB dataset examples contain a lot of junk symbols and tags, so additional
preprocessing is helpful before feeding them into tokenizer.
"""
import re
import html


def spec_add_spaces(t: str) -> str:
    """Add spaces around / and # in `t`."""

    return re.sub(r'([/#])', r' \1 ', t)


def rm_useless_spaces(t: str) -> str:
    """Remove multiple spaces in `t`."""

    return re.sub(' {2,}', ' ', t)


def replace_char_repetitions(t: str, token: str='xxrep') -> str:
    """"Replace repetitions at the character level in `t`."""

    def replace(m) -> str:
        c,cc = m.groups()
        return f' {token} {len(cc)+1} {c} '

    re_rep = re.compile(r'(\S)(\1{3,})')
    return re_rep.sub(replace, t)


def replace_word_repetitions(t: str, token: str='xxwrep') -> str:
    """Replace word repetitions in `t`."""

    def replace(m) -> str:
        c,cc = m.groups()
        return f' {token} {len(cc.split())+1} {c} '

    re_wrep = re.compile(r'(\b\w+\W+)(\1{3,})')
    return re_wrep.sub(replace, t)


def replace_capitalized(t: str, token: str='xxup') -> str:
    """Replace words in all caps in `t`."""

    res = []
    for s in re.findall(r'\w+|\W+', t):
        res += (
            [f' {token} ',s.lower()]
            if s.isupper() and (len(s) > 2)
            else [s.lower()])
    return ''.join(res)


def fix_html(x: str, unknown_token: str='xxunk') -> str:
    """List of replacements from html strings in `x`."""

    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'").replace('amp;', '&').replace(
        '#146;', "'").replace('nbsp;', ' ').replace('#36;', '$').replace(
        '\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace(
        '\\"', '"').replace('<unk>', unknown_token).replace(
        ' @.@ ', '.').replace(' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


# Note that the order of rules matters
default_rules = (
    fix_html,
    replace_char_repetitions,
    replace_word_repetitions,
    replace_capitalized,
    spec_add_spaces,
    rm_useless_spaces,
)