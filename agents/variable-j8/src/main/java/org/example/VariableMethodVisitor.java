package org.example;

import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

import java.util.HashMap;
import java.util.Map;

import org.objectweb.asm.*;


public class VariableMethodVisitor extends MethodVisitor {

    private String methodName;
    private String className;
    private String desc;

    private final Map<Integer, String> localVarNames = new HashMap<>();


    public VariableMethodVisitor(MethodVisitor mv, String name, String className, String desc){
        super(Opcodes.ASM5, mv);
        this.methodName = name;
        this.className = className;
        this.desc = desc;
    }

    // Capture local variable names (only available if compiled with debug info)
    @Override
    public void visitLocalVariable(String name, String descriptor, String signature, Label start, Label end, int index) {
        localVarNames.put(index, name); // Map variable index (slot) to variable name
        super.visitLocalVariable(name, descriptor, signature, start, end, index);
    }

    // Intercept STORE instructions (local variable writes)
    @Override
    public void visitVarInsn(int opcode, int varIndex) {
        if (opcode >= Opcodes.ISTORE && opcode <= Opcodes.ASTORE) {
            String varName = localVarNames.getOrDefault(varIndex, "var" + varIndex); // Fallback to index if no name
            logVariableWrite(varIndex, varName, opcode);
        }
        super.visitVarInsn(opcode, varIndex);
    }

    private void logVariableWrite(int varIndex, String varName, int opcode) {
        // Depending on the type, load the value from the stack and print it
        switch (opcode) {
            case Opcodes.ISTORE: // integer store
                mv.visitVarInsn(Opcodes.ILOAD, varIndex);
                mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Integer", "toString", "(I)Ljava/lang/String;", false);
                break;
            case Opcodes.FSTORE: // float store
                mv.visitVarInsn(Opcodes.FLOAD, varIndex);
                mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Float", "toString", "(F)Ljava/lang/String;", false);
                break;
            case Opcodes.DSTORE: // double store
                mv.visitVarInsn(Opcodes.DLOAD, varIndex);
                mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Double", "toString", "(D)Ljava/lang/String;", false);
                break;
            case Opcodes.LSTORE: // long store
                mv.visitVarInsn(Opcodes.LLOAD, varIndex);
                mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Long", "toString", "(J)Ljava/lang/String;", false);
                break;
            case Opcodes.ASTORE: // reference store
                mv.visitVarInsn(Opcodes.ALOAD, varIndex);
                mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/String", "valueOf", "(Ljava/lang/Object;)Ljava/lang/String;", false);
                break;
        }
        
        // Logging the variable name and value
        mv.visitLdcInsn("Variable " + varName + " = ");
        mv.visitInsn(Opcodes.SWAP);
        mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/System", "out", "(Ljava/lang/Object;)V", false);
    }

}