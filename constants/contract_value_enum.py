import enum

class Contract(enum.Enum):
    LESS_THAN_25K = 'LESS_THAN_25K'
    BTWN_25K_AND_50K = 'BTWN_25K_AND_50K'
    MORE_THAN_50K = 'MORE_THAN_50K'