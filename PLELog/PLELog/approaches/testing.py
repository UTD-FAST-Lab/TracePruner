import sys
sys.path.extend([".", ".."])
from representations.templates.statistics import Simple_template_TF_IDF
import numpy as np
from sklearn.decomposition import FastICA


sequences = []

seq = {
    # 1:'AgentLogger|BRANCH: com.ibm.wala.classLoader.ShrikeBTMethod.isStatic ()Z:ELSE#0',
    # 2:'AgentLogger|BRANCH: com.ibm.wala.classLoader.ShrikeBTMethod.getParameterType (I)Lcom/ibm/wala/types/TypeReference;:IF#0',
    18: 'AgentLogger|PARAMETERS-INFO: null',
    3:'AgentLogger|BRANCH: com.ibm.wala.classLoader.ShrikeBTMethod.getReference ()Lcom/ibm/wala/types/MethodReference;:IF#0',
    4:'AgentLogger|BRANCH: com.ibm.wala.types.MethodReference.getParameterType (I)Lcom/ibm/wala/types/TypeReference;:ELSE#0',
    5:'AgentLogger|BRANCH: com.ibm.wala.types.MethodReference.getParameterType (I)Lcom/ibm/wala/types/TypeReference;:IF#1',
    6:'AgentLogger|BRANCH: com.ibm.wala.ipa.cha.ClassHierarchy.lookupClass (Lcom/ibm/wala/types/TypeReference;)Lcom/ibm/wala/classLoader/IClass;:IF#0',
    # 7:'AgentLogger|BRANCH: com.ibm.wala.ipa.cha.ClassHierarchy.lookupClassRecursive (Lcom/ibm/wala/types/TypeReference;)Lcom/ibm/wala/classLoader/IClass;:ELSE#0',
    8:'AgentLogger|CG_edge: com.ibm.wala.ssa.SSABuilder$SymbolicPropagator.emitInstruction (Lcom/ibm/wala/ssa/SSAInstruction;)V -> com.ibm.wala.ssa.SSABuilder$SymbolicPropagator.getCurrentInstructionIndex ()I',
    9:'AgentLogger|CG_edge: com.ibm.wala.ssa.SSABuilder$SymbolicPropagator.emitInstruction (Lcom/ibm/wala/ssa/SSAInstruction;)V -> com.ibm.wala.ssa.SSAInstruction.getNumberOfDefs ()I',
    10:'AgentLogger|CG_edge: com.ibm.wala.analysis.stackMachine.AbstractIntStackMachine$BasicStackFlowProvider.flow (Lcom/ibm/wala/analysis/stackMachine/AbstractIntStackMachine$MachineState;Lcom/ibm/wala/cfg/ShrikeCFG$BasicBlock;)Lcom/ibm/wala/analysis/stackMachine/AbstractIntStackMachine$MachineState; -> com.ibm.wala.cfg.ShrikeCFG$BasicBlock.getLastInstructionIndex ()I',
    11:'AgentLogger|CG_edge: com.ibm.wala.cfg.ShrikeCFG$BasicBlock.getLastInstructionIndex ()I -> com.ibm.wala.cfg.ShrikeCFG.entry ()Lcom/ibm/wala/cfg/IBasicBlock;',
    15:'AgentLogger|CALL-GRAPH-INFO: TotalNodes=15|CallSite=invokevirtual < Application, Ljava/lang/StringBuilder, append(Ljava/lang/Object;)Ljava/lang/StringBuilder; >@38|Method=Test.main([Ljava/lang/String;)V ; Targets=[]',
    16:'AgentLogger|PARAMETERS-INFO: Param0=[Ljava/lang/StringBuilder] ,',
    12:'AgentLogger|CG_edge: com.ibm.wala.cfg.AbstractCFG.entry ()Lcom/ibm/wala/cfg/IBasicBlock; -> com.ibm.wala.cfg.AbstractCFG.getNode (I)Lcom/ibm/wala/cfg/IBasicBlock;',
    13:'AgentLogger|CG_edge: com.ibm.wala.cfg.AbstractCFG.getNode (I)Lcom/ibm/wala/cfg/IBasicBlock; -> com.ibm.wala.util.graph.impl.DelegatingNumberedNodeManager.getNode (I)Lcom/ibm/wala/util/graph/INodeWithNumber;',
    14:'AgentLogger|CG_edge: com.ibm.wala.cfg.ShrikeCFG$BasicBlock.getLastInstructionIndex ()I -> com.ibm.wala.cfg.ShrikeCFG.exit ()Lcom/ibm/wala/cfg/IBasicBlock;',
    17:'AgentLogger|POINT-TO-MAP-INFO: [Node: < Application, LTest, main([Ljava/lang/String;)V > Context: Everywhere, v1] -> [[Ljava/lang/String] , [Node: < Application, LDeck, <init>()V > Context: Everywhere, v1] -> [LDeck] , [Node: < Application, LDeck, shuffle()LDeck; > Context: Everywhere, v1] -> [LDeck] , [Node: < Application, LHand, <init>(LDeck;)V > Context: Everywhere, v1] -> [LHand] , [Node: < Application, LHand, <init>(LDeck;)V > Context: Everywhere, v2] -> [LDeck] , [Node: < Application, Lcom/google/common/collect/Lists, newArrayList([Ljava/lang/Object;)Ljava/util/ArrayList; > Context: Everywhere, v1] -> [[Ljava/lang/Integer] , [Node: < Application, LHand, draw(LDeck;Ljava/util/List;)LHand; > Context: Everywhere, v1] -> [LHand] , [Node: < Application, LHand, draw(LDeck;Ljava/util/List;)LHand; > Context: Everywhere, v2] -> [LDeck] , [Node: < Application, LDeck, draw()LCard; > Context: Everywhere, v1] -> [LDeck] , [Node: < Application, LDistribution, generate(Ljava/util/Collection;LDeck;)LDistribution; > Context: Everywhere, v2] -> [LDeck] , [Node: < Application, LCard, <init>(LCard$Rank;LCard$Suit;)V > Context: Everywhere, v1] -> [LCard] , [Node: < Application, LHand, <init>(LCard;LCard;LCard;LCard;)V > Context: Everywhere, v1] -> [LHand] , ',
}

seq1 = {
    1:'AgentLogger|BRANCH: com.ibm.wala.classLoader.ShrikeBTMethod.isStatic ()Z:ELSE#0',
    2:'AgentLogger|BRANCH: com.ibm.wala.classLoader.ShrikeBTMethod.getParameterType (I)Lcom/ibm/wala/types/TypeReference;:IF#0',
    18: 'AgentLogger|PARAMETERS-INFO: null',
    3:'AgentLogger|BRANCH: com.ibm.wala.classLoader.ShrikeBTMethod.getReference ()Lcom/ibm/wala/types/MethodReference;:IF#0',
    4:'AgentLogger|BRANCH: com.ibm.wala.types.MethodReference.getParameterType (I)Lcom/ibm/wala/types/TypeReference;:ELSE#0',
    5:'AgentLogger|BRANCH: com.ibm.wala.types.MethodReference.getParameterType (I)Lcom/ibm/wala/types/TypeReference;:IF#1',
    6:'AgentLogger|BRANCH: com.ibm.wala.ipa.cha.ClassHierarchy.lookupClass (Lcom/ibm/wala/types/TypeReference;)Lcom/ibm/wala/classLoader/IClass;:IF#0',
    7:'AgentLogger|BRANCH: com.ibm.wala.ipa.cha.ClassHierarchy.lookupClassRecursive (Lcom/ibm/wala/types/TypeReference;)Lcom/ibm/wala/classLoader/IClass;:ELSE#0',
    8:'AgentLogger|CG_edge: com.ibm.wala.ssa.SSABuilder$SymbolicPropagator.emitInstruction (Lcom/ibm/wala/ssa/SSAInstruction;)V -> com.ibm.wala.ssa.SSABuilder$SymbolicPropagator.getCurrentInstructionIndex ()I',
    9:'AgentLogger|CG_edge: com.ibm.wala.ssa.SSABuilder$SymbolicPropagator.emitInstruction (Lcom/ibm/wala/ssa/SSAInstruction;)V -> com.ibm.wala.ssa.SSAInstruction.getNumberOfDefs ()I',
    10:'AgentLogger|CG_edge: com.ibm.wala.analysis.stackMachine.AbstractIntStackMachine$BasicStackFlowProvider.flow (Lcom/ibm/wala/analysis/stackMachine/AbstractIntStackMachine$MachineState;Lcom/ibm/wala/cfg/ShrikeCFG$BasicBlock;)Lcom/ibm/wala/analysis/stackMachine/AbstractIntStackMachine$MachineState; -> com.ibm.wala.cfg.ShrikeCFG$BasicBlock.getLastInstructionIndex ()I',
    11:'AgentLogger|CG_edge: com.ibm.wala.cfg.ShrikeCFG$BasicBlock.getLastInstructionIndex ()I -> com.ibm.wala.cfg.ShrikeCFG.entry ()Lcom/ibm/wala/cfg/IBasicBlock;',
    15:'AgentLogger|CALL-GRAPH-INFO: TotalNodes=15|CallSite=invokevirtual < Application, Ljava/lang/StringBuilder, append(Ljava/lang/Object;)Ljava/lang/StringBuilder; >@38|Method=Test.main([Ljava/lang/String;)V ; Targets=[]',
    16:'AgentLogger|PARAMETERS-INFO: Param0=[Ljava/lang/StringBuilder] ,',
    12:'AgentLogger|CG_edge: com.ibm.wala.cfg.AbstractCFG.entry ()Lcom/ibm/wala/cfg/IBasicBlock; -> com.ibm.wala.cfg.AbstractCFG.getNode (I)Lcom/ibm/wala/cfg/IBasicBlock;',
    13:'AgentLogger|CG_edge: com.ibm.wala.cfg.AbstractCFG.getNode (I)Lcom/ibm/wala/cfg/IBasicBlock; -> com.ibm.wala.util.graph.impl.DelegatingNumberedNodeManager.getNode (I)Lcom/ibm/wala/util/graph/INodeWithNumber;',
    14:'AgentLogger|CG_edge: com.ibm.wala.cfg.ShrikeCFG$BasicBlock.getLastInstructionIndex ()I -> com.ibm.wala.cfg.ShrikeCFG.exit ()Lcom/ibm/wala/cfg/IBasicBlock;',
    14:'AgentLogger|CG_edge: com.ibm.wala.cfg.ShrikeCFG$BasicBlock.getLastInstructionIndex ()I -> com.ibm.wala.cfg.ShrikeCFG.exit ()Lcom/ibm/wala/cfg/IBasicBlock;',
    14:'AgentLogger|CG_edge: com.ibm.wala.cfg.ShrikeCFG$BasicBlock.getLastInstructionIndex ()I -> com.ibm.wala.cfg.ShrikeCFG.exit ()Lcom/ibm/wala/cfg/IBasicBlock;',
    14:'AgentLogger|CG_edge: com.ibm.wala.cfg.ShrikeCFG$BasicBlock.getLastInstructionIndex ()I -> com.ibm.wala.cfg.ShrikeCFG.exit ()Lcom/ibm/wala/cfg/IBasicBlock;',
    17:'AgentLogger|POINT-TO-MAP-INFO: [Node: < Application, LTest, main([Ljava/lang/String;)V > Context: Everywhere, v1] -> [[Ljava/lang/String] , [Node: < Application, LDeck, <init>()V > Context: Everywhere, v1] -> [LDeck] , [Node: < Application, LDeck, shuffle()LDeck; > Context: Everywhere, v1] -> [LDeck] , [Node: < Application, LHand, <init>(LDeck;)V > Context: Everywhere, v1] -> [LHand] , [Node: < Application, LHand, <init>(LDeck;)V > Context: Everywhere, v2] -> [LDeck] , [Node: < Application, Lcom/google/common/collect/Lists, newArrayList([Ljava/lang/Object;)Ljava/util/ArrayList; > Context: Everywhere, v1] -> [[Ljava/lang/Integer] , [Node: < Application, LHand, draw(LDeck;Ljava/util/List;)LHand; > Context: Everywhere, v1] -> [LHand] , [Node: < Application, LHand, draw(LDeck;Ljava/util/List;)LHand; > Context: Everywhere, v2] -> [LDeck] , [Node: < Application, LDeck, draw()LCard; > Context: Everywhere, v1] -> [LDeck] , [Node: < Application, LDistribution, generate(Ljava/util/Collection;LDeck;)LDistribution; > Context: Everywhere, v2] -> [LDeck] , [Node: < Application, LCard, <init>(LCard$Rank;LCard$Suit;)V > Context: Everywhere, v1] -> [LCard] , [Node: < Application, LHand, <init>(LCard;LCard;LCard;LCard;)V > Context: Everywhere, v1] -> [LHand] , ',
}


sequences.append(seq)
sequences.append(seq1)

seq2label = {1: "true", 2:"false"}





if __name__ == '__main__':

    template =  Simple_template_TF_IDF()

    embdd2label = dict()
    embddings = []

    for indx, seq in enumerate(sequences):

        id_temps = template.present(seq)

        var_info_ids = []
        for id, log in seq.items():
            '''add weights to var info logs in a sequence'''

            if "-INFO" in log:
                var_info_ids.append(id)

        # Number of important sentences and total sentences
        n_imp = len(var_info_ids)
        n_total = len(seq)

        # Avoid division by zero if no important sentences
        if n_imp == 0:
            w_imp = 1.0  # Default to equal weight
        else:
            w_imp = 1 + (n_total / n_imp)  # Adaptive weight for important sentences
        w_non_imp = 1.0  # Default weight for non-important sentences

        # Assign weights dynamically
        weights = {id: w_imp if id in var_info_ids else w_non_imp for id in id_temps}

        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {id: w / total_weight for id, w in weights.items()}

        # Compute weighted sequence embedding
        final_embedding = sum(normalized_weights[id] * np.array(id_temps[id]) for id in id_temps)
        print()
        # print(final_embedding , seq2label[indx+1])  

        embddings.append(final_embedding)
    

        # Convert list to np.array
    embddings = np.asarray(embddings, dtype=np.float64)

    print(embddings)

    # Check for NaNs and Infs
    if np.isnan(embddings).any() or np.isinf(embddings).any():
        print("Warning: Found NaN or Inf in embeddings. Replacing...")
        embddings = np.nan_to_num(embddings, nan=0.0, posinf=1.0, neginf=-1.0)

    # Remove zero-variance columns
    # variances = np.var(embddings, axis=0)
    # if np.sum(variances == 0) > 0:
    #     print("Warning: Removing zero-variance features...")
    #     embddings = embddings[:, variances > 0]

    # print(embddings)

    # Ensure valid n_components
    n_components = min(50, embddings.shape[0])
    print(f"Start FastICA, target dimension: {n_components}")

    # Apply FastICA
    transformer = FastICA(n_components=50)
    train_reprs = transformer.fit_transform(embddings)

    print("Reduced:", train_reprs)
