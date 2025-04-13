

class Instance():

    def __init__(self, program, src, offset, target, is_unknown, edge_id=None, label=None):
        self.program = program
        self.edge_id = edge_id
        self.src = src
        self.target = target
        self.offset = offset
        self.label = label
        self.is_unknown = is_unknown
        self.predicted_label = None
        self.confidence = None
        self.static_features = None
        self.trace_features = None

    def get_id(self):

        if self.edge_id:
            return f'{self.program},{self.edge_id}'
        return f'{self.program},{self.src},{self.target}'
    
    def is_known(self):

        if self.label != None:
            return True
        return False
    
    def get_label(self):

        return self.label
    
    def get_predicted_label(self):

        return self.predicted_label
    
    def set_predicted_label(self, predicted):

        self.predicted_label = predicted

    def set_confidence(self, confidence):

        self.confidence = confidence

    def get_confidence(self):

        return self.confidence
    
    def get_static_featuers(self):

        return self.static_features
    
    def set_static_features(self, static_features):

        self.static_features = static_features

    def get_trace_features(self):

        return self.trace_features
    
    def set_trace_features(self, trace_features):

        self.trace_features = trace_features

