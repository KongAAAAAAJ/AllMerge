"""Payload strata shared by JTP sampling and reports."""

PAYLOAD_GROUPS = ('low', 'medium', 'high')
PAYLOAD_LABELS = dict(zip(PAYLOAD_GROUPS, ('Low load', 'Medium load', 'High load')))
PAYLOAD_KG = dict(zip(PAYLOAD_GROUPS, ((0, 875, 1750), (2625, 3500, 4375), (5250, 6125, 7000))))


def payload_group(payload_kg):
    return {kg: group for group, values in PAYLOAD_KG.items() for kg in values}[float(payload_kg)]
