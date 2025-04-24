package org.example;

import org.objectweb.asm.*;
import org.objectweb.asm.util.CheckClassAdapter;

import org.example.SafeClassWriter;

import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.IllegalClassFormatException;
import java.security.ProtectionDomain;
import java.util.Objects;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
// import java.util.concurrent.ConcurrentHashMap;


public class WalaTraceLoggerTransformer implements ClassFileTransformer  {


        private final WalaTraceLoggerAgent.AgentLevel agentLevel;
        private final WalaTraceLoggerAgent.LogLevel logLevel;
        private final String programPath;

        public WalaTraceLoggerTransformer(WalaTraceLoggerAgent.AgentLevel agentLevel, WalaTraceLoggerAgent.LogLevel logLevel, String programPath) {
            this.agentLevel = agentLevel;
            this.logLevel = logLevel;
            this.programPath = programPath;
        }

        @Override
        public byte[] transform(ClassLoader loader, String className, Class<?> classBeingRedefined,
                                ProtectionDomain protectionDomain, byte[] classfileBuffer) throws IllegalClassFormatException {
            
            if (className == null) {
                return null;
            }
                
            ClassReader cr = new ClassReader(classfileBuffer);
            SafeClassWriter cw = new SafeClassWriter(cr, loader, ClassWriter.COMPUTE_FRAMES);
            ClassVisitor cv = new CheckClassAdapter(cw);
            cv = new WalaTraceLoggerClassVisitor(cv, className, this.agentLevel, this.logLevel, this.programPath);

            cr.accept(cv, ClassReader.EXPAND_FRAMES);
            return cw.toByteArray();
 
        }
}
