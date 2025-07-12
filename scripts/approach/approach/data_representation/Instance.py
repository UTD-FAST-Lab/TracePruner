class Instance():

    def __init__(self, program, src, offset, target, is_unknown, edge_id=None, label=None):
        self.program = program
        self.tool = None
        self.version = None
        self.config_id = None
        self.edge_id = edge_id
        self.src = src
        self.target = target
        self.offset = offset

        self.label = label
        self.ground_truth = None
        self.is_unknown = is_unknown   
             
        self.cluster_label = None
        self.confidence = None
        self.predicted_label = None
        
        self.static_features = None
        self.trace_features = None
        self.var_features = None
        self.semantic_features = None
        self.tokens = None
        self.masks = None


    def get_id(self):

        if self.edge_id:
            return f'{self.program},{self.edge_id}'
        return f'{self.program},{self.src},{self.target}'
    
    def is_known(self):

        if self.is_unknown:
            return False
        return True
    
    def get_label(self):

        return self.label
    
    def set_cluster_label(self, cluster_label):
        self.cluster_label = cluster_label

    def get_cluster_label(self):
        return self.cluster_label
    
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

    def get_var_features(self):
        return self.var_features
    
    def set_var_features(self, var_features):
        self.var_features = var_features

    def get_semantic_features(self):
        return self.semantic_features
    
    def set_semantic_features(self, semantic_features):
        self.semantic_features = semantic_features

    def set_ground_truth(self, value):
        self.ground_truth = value

    def get_ground_truth(self):
        return self.ground_truth
    

    def set_trace_graph(self, trace_graph):
        self.trace_graph = trace_graph

    def get_trace_graph(self):
        return self.trace_graph
    
    def set_tokens(self, tokens):
        self.tokens = tokens

    def get_tokens(self):
        return self.tokens
    
    def set_masks(self, masks):
        self.masks = masks
        
    def get_masks(self):
        return self.masks