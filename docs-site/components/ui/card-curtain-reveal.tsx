"use client";

import * as React from "react";
import { type HTMLMotionProps, type Variants, motion } from "motion/react";
import { cn } from "cn";

const curtainVariants: Variants = {
  visible: {
    clipPath: "polygon(0 0,100% 0,100% 100%,0 100%)",
    transition: { duration: 0.4, ease: [0.25, 1.5, 0.5, 1] },
  },
  hidden: {
    clipPath: "polygon(50% 0,50% 0,50% 100%,50% 100%)",
    transition: { duration: 0.3, ease: [0.25, 1.5, 0.5, 1] },
  },
};

interface CardCurtainRevealContextValue {
  revealed: boolean;
  /** False on touch-only devices: details stay visible and the curtain never draws. */
  canHover: boolean;
}
const CardCurtainRevealContext = React.createContext<CardCurtainRevealContextValue | undefined>(undefined);
function useCardCurtainRevealContext() {
  const context = React.useContext(CardCurtainRevealContext);
  if (!context) throw new Error("useCardCurtainRevealContext must be used within a CardCurtainReveal component");
  return context;
}

function useCanHover() {
  const [canHover, setCanHover] = React.useState(true);
  React.useEffect(() => {
    const query = window.matchMedia("(hover: hover) and (pointer: fine)");
    const sync = () => setCanHover(query.matches);
    sync();
    query.addEventListener("change", sync);
    return () => query.removeEventListener("change", sync);
  }, []);
  return canHover;
}

interface CardCurtainRevealProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Reveals the card without a pointer, for example while its link has keyboard focus. */
  active?: boolean;
}

const CardCurtainReveal = React.forwardRef<HTMLDivElement, CardCurtainRevealProps>(
  ({ children, className, active = false, ...props }, ref) => {
    const [isMouseIn, setIsMouseIn] = React.useState(false);
    const canHover = useCanHover();
    const handleMouseEnter = React.useCallback(() => setIsMouseIn(true), []);
    const handleMouseLeave = React.useCallback(() => setIsMouseIn(false), []);
    const value = React.useMemo(
      () => ({ revealed: !canHover || isMouseIn || active, canHover }),
      [active, canHover, isMouseIn],
    );
    return (
      <CardCurtainRevealContext.Provider value={value}>
        <div
          ref={ref}
          className={cn("relative flex flex-col gap-2 overflow-hidden", className)}
          data-revealed={value.revealed || undefined}
          data-curtain-open={(canHover && value.revealed) || undefined}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
          {...props}
        >
          {children}
        </div>
      </CardCurtainRevealContext.Provider>
    );
  },
);
CardCurtainReveal.displayName = "CardCurtainReveal";

const CardCurtainRevealFooter = React.forwardRef<HTMLDivElement, HTMLMotionProps<"div">>(
  ({ className, ...props }, ref) => {
    const { revealed } = useCardCurtainRevealContext();
    return (
      <motion.div
        ref={ref}
        className={className}
        variants={curtainVariants}
        initial={false}
        animate={revealed ? "visible" : "hidden"}
        {...props}
      />
    );
  },
);
CardCurtainRevealFooter.displayName = "CardCurtainRevealFooter";

const CardCurtainRevealBody = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => <div ref={ref} className={cn("flex-1 p-6", className)} {...props} />,
);
CardCurtainRevealBody.displayName = "CardCurtainRevealBody";

interface CardCurtainRevealTitleProps extends HTMLMotionProps<"h2"> {
  /** Distance in pixels the title rests below its revealed position. */
  hiddenOffset?: number;
  as?: "h2" | "h3" | "h4";
}

const CardCurtainRevealTitle = React.forwardRef<HTMLHeadingElement, CardCurtainRevealTitleProps>(
  ({ className, hiddenOffset = 170, as = "h2", ...props }, ref) => {
    const { revealed } = useCardCurtainRevealContext();
    const Heading = motion[as];
    return (
      <Heading
        ref={ref}
        className={className}
        initial={false}
        animate={{ y: revealed ? 0 : hiddenOffset }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        {...props}
      />
    );
  },
);
CardCurtainRevealTitle.displayName = "CardCurtainRevealTitle";

const CardCurtain = React.forwardRef<HTMLDivElement, HTMLMotionProps<"div">>(({ className, ...props }, ref) => {
  const { revealed, canHover } = useCardCurtainRevealContext();
  if (!canHover) return null;
  return (
    <motion.div
      ref={ref}
      aria-hidden="true"
      className={cn("pointer-events-none absolute inset-0 size-full mix-blend-difference", className)}
      variants={curtainVariants}
      initial={false}
      animate={revealed ? "visible" : "hidden"}
      {...props}
    />
  );
});
CardCurtain.displayName = "CardCurtain";

const CardCurtainRevealDescription = React.forwardRef<HTMLDivElement, HTMLMotionProps<"div">>(
  ({ className, ...props }, ref) => {
    const { revealed } = useCardCurtainRevealContext();
    return (
      <motion.div
        ref={ref}
        className={className}
        variants={curtainVariants}
        initial={false}
        animate={revealed ? "visible" : "hidden"}
        {...props}
      />
    );
  },
);
CardCurtainRevealDescription.displayName = "CardCurtainRevealDescription";

export {
  CardCurtainReveal,
  CardCurtainRevealBody,
  CardCurtainRevealFooter,
  CardCurtainRevealDescription,
  CardCurtainRevealTitle,
  CardCurtain,
};
