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
    // Call the original instruction first
        super.visitVarInsn(opcode, varIndex);

        if (opcode >= Opcodes.ISTORE && opcode <= Opcodes.ASTORE) {
            // Get the variable name for logging
            String varName = localVarNames.getOrDefault(varIndex, "var" + varIndex);

            // Load System.out
            mv.visitFieldInsn(Opcodes.GETSTATIC, "java/lang/System", "out", "Ljava/io/PrintStream;");

            // Create StringBuilder
            mv.visitTypeInsn(Opcodes.NEW, "java/lang/StringBuilder");
            mv.visitInsn(Opcodes.DUP);
            mv.visitMethodInsn(Opcodes.INVOKESPECIAL, "java/lang/StringBuilder", "<init>", "()V", false);

            // Append "Variable <varName> = "
            mv.visitLdcInsn("AgentLogger|Variable: " + className.replace('/', '.') + "." + methodName + " " + desc +" Variable " + varName + " = ");
            mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, "java/lang/StringBuilder", "append", "(Ljava/lang/String;)Ljava/lang/StringBuilder;", false);

            // Load the variable value and append it
            switch (opcode) {
                case Opcodes.ISTORE:
                    mv.visitVarInsn(Opcodes.ILOAD, varIndex);
                    mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, "java/lang/StringBuilder", "append", "(I)Ljava/lang/StringBuilder;", false);
                    break;
                case Opcodes.FSTORE:
                    mv.visitVarInsn(Opcodes.FLOAD, varIndex);
                    mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, "java/lang/StringBuilder", "append", "(F)Ljava/lang/StringBuilder;", false);
                    break;
                case Opcodes.DSTORE:
                    mv.visitVarInsn(Opcodes.DLOAD, varIndex);
                    mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, "java/lang/StringBuilder", "append", "(D)Ljava/lang/StringBuilder;", false);
                    break;
                case Opcodes.LSTORE:
                    mv.visitVarInsn(Opcodes.LLOAD, varIndex);
                    mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, "java/lang/StringBuilder", "append", "(J)Ljava/lang/StringBuilder;", false);
                    break;
                case Opcodes.ASTORE:
                    mv.visitVarInsn(Opcodes.ALOAD, varIndex);
                    mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, "java/lang/StringBuilder", "append", "(Ljava/lang/Object;)Ljava/lang/StringBuilder;", false);
                    break;
            }

            // Convert StringBuilder to String and print
            mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, "java/lang/StringBuilder", "toString", "()Ljava/lang/String;", false);
            mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, "java/io/PrintStream", "println", "(Ljava/lang/String;)V", false);
        }

    }
    
}