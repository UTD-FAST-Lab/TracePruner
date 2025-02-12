package org.example;

import org.objectweb.asm.*;

import org.example.SafeClassWriter;

import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.IllegalClassFormatException;
import java.security.ProtectionDomain;
import java.util.Objects;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class IntegratedLoggerTransformer implements ClassFileTransformer  {


        private final IntegratedLoggerAgent.AgentLevel agentLevel;
        private final IntegratedLoggerAgent.LogLevel logLevel;

        private final Set<String> exclusion = new HashSet<>(Arrays.asList(
            "com/ibm/wala/util/intset/BitVectorIntSet",
            "com/ibm/wala/util/intset/SemiSparseMutableIntSet",
            "com/ibm/wala/cfg/ShrikeCFG$BasicBlock"
        ));



        public IntegratedLoggerTransformer(IntegratedLoggerAgent.AgentLevel agentLevel, IntegratedLoggerAgent.LogLevel logLevel){
            this.agentLevel = agentLevel;
            this.logLevel = logLevel;
        }

        private boolean isExcluded(String className) {
            return exclusion.contains(className);
        }

        @Override
        public byte[] transform(ClassLoader loader, String className, Class<?> classBeingRedefined,
                                ProtectionDomain protectionDomain, byte[] classfileBuffer) throws IllegalClassFormatException {
            
            // if (className == null || isExcluded(className)) {
            if (className == null) {
                return null;
            }
                
            // System.out.println("Transforming class: " + className);
            ClassReader cr = new ClassReader(classfileBuffer);
            SafeClassWriter cw = new SafeClassWriter(cr, loader, ClassWriter.COMPUTE_FRAMES);
            // ClassWriter cw = new ClassWriter(cr, ClassWriter.COMPUTE_FRAMES);
            ClassVisitor cv = new IntegratedClassVisitor(cw, className, this.agentLevel, this.logLevel);
            cr.accept(cv, ClassReader.EXPAND_FRAMES);
            return cw.toByteArray();
            
            // return null;
        }
}
