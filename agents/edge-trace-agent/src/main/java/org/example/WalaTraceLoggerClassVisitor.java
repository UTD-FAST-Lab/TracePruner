package org.example;

import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

public class WalaTraceLoggerClassVisitor extends ClassVisitor {

    private String className;
    private final WalaTraceLoggerAgent.AgentLevel agentLevel;
    private final WalaTraceLoggerAgent.LogLevel logLevel;
    private final String programPath;

    public WalaTraceLoggerClassVisitor(ClassVisitor cv, String className, WalaTraceLoggerAgent.AgentLevel agentLevel, WalaTraceLoggerAgent.LogLevel logLevel, String programPath) {
        super(Opcodes.ASM5, cv);
        this.className = className;
        this.agentLevel = agentLevel;
        this.logLevel = logLevel;
        this.programPath = programPath;
    }

    @Override
    public MethodVisitor visitMethod(int access, String name, String desc, String signature, String[] exceptions){
        MethodVisitor mv = super.visitMethod(access, name, desc, signature, exceptions);
        return new WalaTraceLoggerMethodVisitor(mv, name, className, desc, agentLevel, logLevel, programPath);
    }



}
