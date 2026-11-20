"""Reproducible scene selection and consistent method-by-scene visualizations."""
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from evaluation.ablation.anchor_ablation.data import select_figure_samples
from evaluation.ablation.anchor_ablation.figure_selection import (
    sample_figure_candidates, select_safety_improvement_scenes,
)
from evaluation.ablation.anchor_ablation.runner import _anchor_pass
from evaluation.ablation.anchor_ablation.sources import AnchorSource
from evaluation.ablation.anchor_ablation.visualization import draw_anchor_ablation, draw_diffusion_planning


def test_selects_distinct_episodes_reproducibly_without_duplicate_frames():
    metadata = [dict(episode_index=e, frame_index=f, ego_index=g)
                for e in range(5) for f in range(4) for g in range(3)]
    selected = select_figure_samples(metadata, count=3, seed=2027)
    assert selected == select_figure_samples(list(reversed(metadata)), count=3, seed=2027)
    assert len({key[0] for key in selected}) == 3
    assert len({key[:2] for key in selected}) == 3


def test_falls_back_to_distinct_frames_and_never_duplicates_short_datasets():
    metadata = [dict(episode_index=0, frame_index=f, ego_index=g)
                for f in range(2) for g in range(3)]
    assert len(select_figure_samples(metadata, count=3)) == 2
    with pytest.raises(ValueError):
        select_figure_samples(metadata, count=0)


def test_anchor_pass_collects_all_selected_scenes_and_preserves_sample_count():
    b = 4
    xy = torch.zeros(b, 10, 8, 2)
    xy[..., 0] = torch.arange(1, 9)
    xy[..., 1] = torch.arange(10)[None, :, None] * .2
    features = {
        'coarse_trajectories': xy, 'coarse_controls': torch.zeros(b, 10, 40, 2),
        'mode_valid_mask': torch.ones(b, 10, dtype=torch.bool),
        'ego_state': torch.zeros(b, 14), 'agent_states': torch.zeros(b, 2, 11),
        'agent_valid_mask': torch.zeros(b, 2, dtype=torch.bool),
        'map_polylines': torch.zeros(b, 1, 8, 10),
        'map_valid_mask': torch.ones(b, 1, dtype=torch.bool),
        'target_point': torch.zeros(b, 2),
        'road_lane_geometry': torch.zeros(b, 1, 8, 9),
        'road_lane_point_mask': torch.ones(b, 1, 8, dtype=torch.bool),
        'road_lane_mask': torch.ones(b, 1, dtype=torch.bool),
    }
    metadata = [dict(episode_index=e, frame_index=0, ego_index=0) for e in range(b)]
    keys = select_figure_samples(metadata, count=3)
    class SingleBatch(list):
        batch_size = b
        dataset = range(b)

    loader = SingleBatch([(features, {'trajectory': xy[:, 0]}, metadata)])
    with patch('evaluation.ablation.anchor_ablation.runner.anchor_safety_cost',
               return_value={'hard_safe': torch.ones(b, 10, dtype=torch.bool),
                             'collision_unsafe': torch.zeros(b, 10, dtype=torch.bool),
                             'road_unsafe': torch.zeros(b, 10, dtype=torch.bool)}):
        rows, figures = _anchor_pass('A1', AnchorSource({'strategy': 'dynamic'}),
                                    loader, torch.device('cpu'), {}, 10, keys)
    assert len(rows) == b
    assert set(figures) == set(keys)
    for key, item in figures.items():
        np.testing.assert_allclose(item['anchors_xy'], xy[key[0]].numpy())
        assert item['safety']['unsafe_count'] == 0
        assert item['hard_safe'].all()
    # Enlarging the figure pool never changes the quantitative sample rows.
    with patch('evaluation.ablation.anchor_ablation.runner.anchor_safety_cost',
               return_value={'hard_safe': torch.ones(b, 10, dtype=torch.bool),
                             'collision_unsafe': torch.zeros(b, 10, dtype=torch.bool),
                             'road_unsafe': torch.zeros(b, 10, dtype=torch.bool)}):
        more_rows, more_figures = _anchor_pass('A1', AnchorSource({'strategy': 'dynamic'}),
                                              loader, torch.device('cpu'), {}, 10,
                                              [(e, 0, 0) for e in range(b)])
    assert more_rows == rows
    assert len(more_figures) == b


def test_candidate_sampling_is_reproducible_and_contains_multiple_egos():
    metadata = [dict(episode_index=e, frame_index=f, ego_index=g)
                for e in range(3) for f in range(4) for g in range(3)]
    keys = sample_figure_candidates(metadata, count=20, seed=7)
    assert keys == sample_figure_candidates(metadata[::-1], count=20, seed=7)
    assert len(keys) == len(set(keys)) == 20
    assert len(sample_figure_candidates(metadata, count=100, seed=7)) == 36


def _scene_safety(unsafe, road, collision):
    return {'safety': {'valid_count': 10, 'safe_count': 10 - unsafe,
                       'unsafe_count': unsafe, 'road_unsafe_count': road,
                       'collision_unsafe_count': collision}}


def test_safety_ranking_uses_union_count_and_distinct_frames_and_episodes():
    # Key 0 has fewer unsafe anchors overall despite overlapping road/collision
    # labels. Key 1 has the larger primary gap and must rank ahead of it.
    keys = [(0, 1, 0), (1, 1, 0), (2, 1, 0), (1, 1, 1)]
    a1 = [_scene_safety(5, 5, 5), _scene_safety(7, 4, 3),
          _scene_safety(4, 3, 1), _scene_safety(8, 4, 4)]
    a2 = [_scene_safety(2, 1, 1), _scene_safety(2, 1, 1),
          _scene_safety(2, 1, 1), _scene_safety(1, 1, 0)]
    payloads = {'A1': dict(zip(keys, a1)), 'A2': dict(zip(keys, a2))}
    selected, ranking = select_safety_improvement_scenes(payloads, count=3)
    assert selected == [(1, 1, 1), (0, 1, 0), (2, 1, 0)]
    assert [row['unsafe_anchor_reduction'] for row in ranking] == [7, 5, 3, 2]
    assert sum(row['selected'] for row in ranking) == 3
    assert len(payloads['A1']) == 4  # selection does not mutate evaluated scenes


def test_positive_scenes_are_not_replaced_by_negative_scenes_for_episode_variety():
    keys = [(0, 1, 0), (0, 2, 0), (1, 1, 0)]
    payloads = {'A1': {key: _scene_safety(n, n, 0) for key, n in zip(keys, [8, 6, 1])},
                'A2': {key: _scene_safety(2, 2, 0) for key in keys}}
    selected, _ = select_safety_improvement_scenes(payloads, count=2)
    assert selected == keys[:2]
    selected, ranking = select_safety_improvement_scenes(payloads, count=4)
    assert len(selected) == 3
    assert ranking[-1]['unsafe_anchor_reduction'] == -1


@pytest.mark.parametrize('method_count', [1, 2, 3])
@pytest.mark.parametrize('figure_kind', ['anchors', 'planning'])
def test_grid_has_one_row_per_scene_and_one_column_per_method(tmp_path, method_count, figure_kind):
    item = {
        'features': {'map_polylines': np.zeros((1, 8, 10)),
                     'map_valid_mask': np.ones(1, dtype=bool),
                     'agent_states': np.zeros((1, 11)),
                     'agent_valid_mask': np.zeros(1, dtype=bool),
                     'target_point': np.array([30., 0.])},
        'expert_xy': np.column_stack([np.arange(1, 9) * 5, np.zeros(8)]),
        'anchors_xy': np.zeros((10, 8, 2)), 'valid': np.ones(10, dtype=bool),
        'hard_safe': np.ones(10, dtype=bool),
    }
    item.update(diffusion_candidates=item['anchors_xy'], diffusion_valid=item['valid'],
                selected_raw_xy=item['expert_xy'])
    methods = ('A0', 'A1', 'A2')[:method_count]
    scenes = [{m: item for m in methods} for _ in range(3)]
    original_close = plt.close
    captured = []

    def check_close(fig):
        if hasattr(fig, 'axes'):
            captured.append(len(fig.axes))
            assert len(fig.axes) == 3 * method_count
            assert len(fig.legends) == 1 and not fig.legends[0].get_frame_on()
            from matplotlib.text import Text
            assert all(text.get_fontfamily() == ['Times New Roman'] for text in fig.findobj(Text))
            if method_count > 1:
                assert fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[1])
            assert not fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[method_count])
            labels = [text.get_text() for text in fig.texts]
            assert labels.count('Longitudinal position (m)') == 1
            assert labels.count('Lateral position (m)') == 1
            titles = ('Fixed Anchors', 'Dynamic Anchors', 'Ours')[:method_count]
            for index, ax in enumerate(fig.axes):
                assert ax.get_title() == (f'Example {index // method_count + 1}'
                                          if index % method_count == 0 else '')
                tags = [text for text in ax.texts if text.get_text() == titles[index % method_count]]
                assert len(tags) == 1
                assert tags[0].get_transform() is ax.transAxes
                box = tags[0].get_bbox_patch()
                assert 0 < box.get_facecolor()[3] < 1
                assert box.get_linewidth() > 0
                assert ax.get_xlabel() == ax.get_ylabel() == ''
                if figure_kind == 'planning':
                    assert not ax.child_axes
                    selected = [line for line in ax.lines
                                if line.get_label() == 'Selected diffusion trajectory'][0]
                    assert selected.get_color() == '#d62728' and selected.get_linestyle() == ':'
                for panel in [ax, *ax.child_axes]:
                    assert 'Target' not in panel.get_legend_handles_labels()[1]
                    expert = [line for line in panel.lines if line.get_label() == 'Expert trajectory']
                    assert len(expert) == 1 and expert[0].get_linewidth() == 0.9
                if index % method_count == 0:
                    assert any(label.get_visible() for label in ax.get_xticklabels())
                else:
                    assert not ax.get_xticklabels()
        original_close(fig)

    with patch.object(plt, 'close', side_effect=check_close):
        path = tmp_path / f'{figure_kind}.pdf'
        draw = draw_anchor_ablation if figure_kind == 'anchors' else draw_diffusion_planning
        draw(path, scenes, scene_labels=['Example 1', 'Example 2', 'Example 3'])
        assert path.stat().st_size > 0
    assert captured == [3 * method_count]


def test_unsafe_anchor_styles_follow_original_indices_after_validity_filtering(tmp_path):
    from matplotlib.colors import to_rgba
    from evaluation.ablation.anchor_ablation import visualization

    x = np.arange(1, 9) * 5.0
    anchors = np.stack([np.column_stack((x, np.full(8, i))) for i in range(4)])
    item = {
        'features': {'map_polylines': np.zeros((1, 8, 10)),
                     'map_valid_mask': np.ones(1, dtype=bool),
                     'agent_states': np.zeros((1, 11)),
                     'agent_valid_mask': np.zeros(1, dtype=bool),
                     'target_point': np.array([30., 0.])},
        'expert_xy': anchors[0], 'anchors_xy': anchors,
        'valid': np.array([True, False, True, True]),
        'hard_safe': np.array([True, False, False, True]),
    }
    original_finish = visualization._finish_anchor_grid

    def check_styles(fig, grid, path, sample_label):
        fills = []
        for method, ax in zip(('A0', 'A1', 'A2'), grid[0]):
            curves = [line for line in ax.lines if line.get_zorder() == 2]
            assert len(curves) == 3
            assert [line.get_linestyle() for line in curves] == (['-', '--', '-']
                                                                 if method != 'A2' else ['-', '-', '-'])
            np.testing.assert_array_equal(curves[1].get_ydata()[1:], anchors[2, :, 1])
            if method != 'A2':
                assert to_rgba(curves[1].get_color()) == to_rgba(visualization.UNSAFE_ANCHOR_COLOR)
                assert curves[1].get_alpha() == visualization.UNSAFE_ANCHOR_ALPHA
            assert curves[0].get_alpha() == 0.68
            assert curves[2].get_color() == plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
            fills.append(ax.texts[0].get_bbox_patch().get_facecolor())
        assert len(set(fills)) == 3
        original_finish(fig, grid, path, sample_label)

    with patch.object(visualization, '_finish_anchor_grid', side_effect=check_styles):
        draw_anchor_ablation(tmp_path / 'safety_styles.pdf', {m: item for m in ('A0', 'A1', 'A2')})


def test_insets_share_row_limits_and_support_precise_position_overrides(tmp_path, monkeypatch):
    from evaluation.ablation.anchor_ablation import visualization

    x = np.linspace(5, 50, 8)
    item = {
        'features': {'map_polylines': np.zeros((1, 8, 10)),
                     'map_valid_mask': np.ones(1, dtype=bool),
                     'agent_states': np.zeros((1, 11)),
                     'agent_valid_mask': np.zeros(1, dtype=bool)},
        'expert_xy': np.column_stack((x, x * 0.1)),
        'anchors_xy': np.stack([np.column_stack((x, x * slope)) for slope in (0.05, 0.2)]),
        'valid': np.ones(2, dtype=bool), 'hard_safe': np.ones(2, dtype=bool),
    }
    monkeypatch.setattr(visualization, 'ANCHOR_INSET_ROW_LIMITS',
                        {0: {'xlim': (12, 35), 'ylim': (-1, 8)}})
    bounds = (0.04, 0.08, 0.52, 0.50)
    monkeypatch.setattr(visualization, 'ANCHOR_INSET_OVERRIDES',
                        {(0, 'A1'): {'bounds': bounds}, (1, 'A2'): {'enabled': False}})
    original_close = plt.close

    def check_close(fig):
        if hasattr(fig, 'axes'):
            first_a1, first_a2 = fig.axes[1].child_axes[0], fig.axes[2].child_axes[0]
            assert first_a1.get_xlim() == first_a2.get_xlim() == (12, 35)
            assert first_a1.get_ylim() == first_a2.get_ylim() == (-1, 8)
            assert len(fig.axes[0].child_axes) == 0
            assert len(fig.axes[4].child_axes) == 1
            assert len(fig.axes[5].child_axes) == 0
            parent = fig.axes[1].get_position()
            child = first_a1.get_position()
            actual = ((child.x0-parent.x0)/parent.width, (child.y0-parent.y0)/parent.height,
                      child.width/parent.width, child.height/parent.height)
            np.testing.assert_allclose(actual, bounds)
        original_close(fig)

    with patch.object(plt, 'close', side_effect=check_close):
        draw_anchor_ablation(tmp_path / 'insets.pdf', [{m: item for m in ('A0', 'A1', 'A2')}] * 2)
