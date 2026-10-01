class Personality:
    def __init__(self, honesty, courage, intelligence, kindness, suspicion, confidence):
        self.honesty = self._validate_value(honesty)
        self.courage = self._validate_value(courage)
        self.intelligence = self._validate_value(intelligence)
        self.kindness = self._validate_value(kindness)
        self.suspicion = self._validate_value(suspicion)
        self.confidence = self._validate_value(confidence)

    def _validate_value(self, value):
        if value < 0 or value > 1:
            raise ValueError("Personality value must be between 0 and 1")

        return value
    


    