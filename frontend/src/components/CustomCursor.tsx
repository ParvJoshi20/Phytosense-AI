'use client';

import React, { useEffect, useState } from 'react';
import { motion, useMotionValue, useSpring } from 'framer-motion';

interface CustomCursorProps {
  statusColor?: string; // e.g. '#2A7FFF' (Scanner Blue) or '#FFB020' (Alert Amber)
}

export const CustomCursor: React.FC<CustomCursorProps> = ({
  statusColor = '#2A7FFF',
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [isPointer, setIsPointer] = useState(false);

  const mouseX = useMotionValue(-100);
  const mouseY = useMotionValue(-100);

  const springConfig = { damping: 28, stiffness: 350, mass: 0.4 };
  const cursorX = useSpring(mouseX, springConfig);
  const cursorY = useSpring(mouseY, springConfig);

  // Trailing ambient light spring for soft chiaroscuro background glow
  const ambientSpringConfig = { damping: 35, stiffness: 180, mass: 0.8 };
  const ambientX = useSpring(mouseX, ambientSpringConfig);
  const ambientY = useSpring(mouseY, ambientSpringConfig);

  useEffect(() => {
    // Only enable on fine pointer devices (desktop/mouse)
    if (typeof window === 'undefined' || !window.matchMedia('(pointer: fine)').matches) {
      return;
    }

    const handleMouseMove = (e: MouseEvent) => {
      mouseX.set(e.clientX);
      mouseY.set(e.clientY);
      if (!isVisible) setIsVisible(true);

      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'BUTTON' ||
          target.tagName === 'A' ||
          target.tagName === 'INPUT' ||
          target.closest('button') ||
          target.closest('a') ||
          target.getAttribute('role') === 'button' ||
          window.getComputedStyle(target).cursor === 'pointer')
      ) {
        setIsPointer(true);
      } else {
        setIsPointer(false);
      }
    };

    const handleMouseLeave = () => setIsVisible(false);
    const handleMouseEnter = () => setIsVisible(true);

    window.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseleave', handleMouseLeave);
    document.addEventListener('mouseenter', handleMouseEnter);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseleave', handleMouseLeave);
      document.removeEventListener('mouseenter', handleMouseEnter);
    };
  }, [mouseX, mouseY, isVisible]);

  if (!isVisible) return null;

  return (
    <div className="pointer-events-none fixed inset-0 z-50 overflow-hidden">
      {/* Dynamic Ambient Light Source illuminating dark UI */}
      <motion.div
        style={{
          x: ambientX,
          y: ambientY,
          translateX: '-50%',
          translateY: '-50%',
        }}
        className="absolute w-80 h-80 rounded-full opacity-15 blur-3xl pointer-events-none transition-colors duration-700"
        animate={{
          backgroundColor: statusColor,
        }}
      />

      {/* Trailing Outer Ring */}
      <motion.div
        style={{
          x: cursorX,
          y: cursorY,
          translateX: '-50%',
          translateY: '-50%',
        }}
        className="absolute rounded-full border pointer-events-none flex items-center justify-center transition-colors duration-300"
        animate={{
          width: isPointer ? 36 : 24,
          height: isPointer ? 36 : 24,
          borderColor: statusColor,
          backgroundColor: isPointer ? `${statusColor}18` : 'transparent',
          boxShadow: `0 0 15px ${statusColor}66`,
        }}
        transition={{ type: 'spring', damping: 25, stiffness: 300 }}
      />

      {/* Sharp Glowing Cursor Tip */}
      <motion.div
        style={{
          x: mouseX,
          y: mouseY,
          translateX: '-50%',
          translateY: '-50%',
        }}
        className="absolute w-2 h-2 rounded-full pointer-events-none transition-colors duration-300 shadow-md"
        animate={{
          backgroundColor: statusColor,
          boxShadow: `0 0 10px ${statusColor}, 0 0 20px ${statusColor}`,
          scale: isPointer ? 1.5 : 1,
        }}
      />
    </div>
  );
};
