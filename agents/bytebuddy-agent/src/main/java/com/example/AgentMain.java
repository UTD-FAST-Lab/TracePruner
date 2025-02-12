import net.bytebuddy.agent.builder.AgentBuilder;
import net.bytebuddy.asm.Advice;
import net.bytebuddy.description.type.TypeDescription;
import net.bytebuddy.dynamic.loading.ClassReloadingStrategy;
import net.bytebuddy.matcher.ElementMatchers;

import java.lang.instrument.Instrumentation;

public class AgentMain {
    public static boolean isPrintingEnabled = false;

    public static void premain(String arguments, Instrumentation instrumentation) {
        new AgentBuilder.Default()
                .type(ElementMatchers.any()) // Instrument all loaded classes
                .transform((builder, typeDescription, classLoader, module) -> builder
                        .visit(Advice.to(StartAdvice.class).on(ElementMatchers.named("makeZeroCFABuilder"))) // Replace "startMethod" with your method name
                        .visit(Advice.to(EndAdvice.class).on(ElementMatchers.named("makeCallGraph"))) // Replace "endMethod" with your method name
                        .visit(Advice.to(MethodCallLogger.class).on(ElementMatchers.any())) // Log method calls only if enabled
                )
                .installOn(instrumentation);
    }

    // Advice for "startMethod"
    public static class StartAdvice {
        @Advice.OnMethodEnter
        public static void onEnter() {
            isPrintingEnabled = true;
            System.out.println("Printing started.");
        }
    }

    // Advice for "endMethod"
    public static class EndAdvice {
        @Advice.OnMethodExit
        public static void onExit() {
            isPrintingEnabled = false;
            System.out.println("Printing stopped.");
        }
    }

    // General logging advice (conditionally logs method calls)
    public static class MethodCallLogger {
        @Advice.OnMethodEnter
        public static void onMethodEnter(@Advice.Origin("#t #m") String method) {
            if (isPrintingEnabled) {
                System.out.println("Method invoked: " + method);
            }
        }
    }
}
