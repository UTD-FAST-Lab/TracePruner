package org.example;

import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

public class IntegratedClassVisitor extends ClassVisitor {

    private String className;
    private final IntegratedLoggerAgent.AgentLevel agentLevel;
    private final IntegratedLoggerAgent.LogLevel logLevel;

    public IntegratedClassVisitor(ClassVisitor cv, String className, IntegratedLoggerAgent.AgentLevel agentLevel, IntegratedLoggerAgent.LogLevel logLevel){
        super(Opcodes.ASM5, cv);
        this.className = className;
        this.agentLevel = agentLevel;
        this.logLevel = logLevel;

    }

    @Override
    public MethodVisitor visitMethod(int access, String name, String desc, String signature, String[] exceptions){
        MethodVisitor mv = super.visitMethod(access, name, desc, signature, exceptions);
        return new IntegratedMethodVisitor(mv, name, className, desc, agentLevel, logLevel);
    }



}
