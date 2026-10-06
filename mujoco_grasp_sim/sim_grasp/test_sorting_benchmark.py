"""SP3 batch summaries keep wrong-bin counts and failed seeds visible."""
from benchmark import summarize_sorting, format_location_agreement


rows = [
    {'seed': 0, 'crashed': False, 'total': 4, 'in_correct_bin': 2,
     'in_wrong_bin': 1, 'in_bin': 3, 'fell_off': 0},
    {'seed': 1, 'crashed': True, 'error_tail': 'NIM unavailable'},
    {'seed': 2, 'crashed': False, 'total': 4, 'in_correct_bin': 3,
     'in_wrong_bin': 0, 'in_bin': 3, 'fell_off': 1},
]
summary = summarize_sorting(rows)
assert summary == {'completed': 2, 'crashed': 1, 'total': 8,
                   'correct': 5, 'wrong': 1, 'binned': 6, 'fell_off': 1}
assert format_location_agreement([1.0] * 9 + [0.958]) == '99.6%'

print('SP3 benchmark aggregation checks passed.')
