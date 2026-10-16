import React, { useState, useEffect, useRef } from 'react';
import './streaming-text.css';

/**
 * StreamingText Component
 * Progressive streaming text renderer for AI chat and agent interfaces.
 * Compatible with @kobra/streaming-text registry specification.
 */
export function StreamingText({
  content = '',
  speed = 12,
  cursor = true,
  cursorChar = '▊',
  fade = true,
  immediate = false,
  onComplete,
  className = '',
  dataSlot = 'streaming-text',
}) {
  const [displayedLength, setDisplayedLength] = useState(immediate ? content.length : 0);
  const onCompleteRef = useRef(onComplete);

  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  useEffect(() => {
    if (immediate) return;

    if (displayedLength >= content.length) {
      onCompleteRef.current?.();
      return;
    }

    // Natural token cadence: 2-3 characters per tick
    const timer = setTimeout(() => {
      const step = Math.min(3, content.length - displayedLength);
      setDisplayedLength((prev) => prev + step);
    }, speed);

    return () => clearTimeout(timer);
  }, [content, displayedLength, speed, immediate]);

  const isDone = immediate || displayedLength >= content.length;
  const visibleText = immediate ? content : content.slice(0, displayedLength);

  return (
    <div
      className={`streaming-text-container ${className} ${fade ? 'fade-stream' : ''}`}
      data-slot={dataSlot}
    >
      <FormattedContent text={visibleText} />
      {!isDone && cursor && (
        <span className="streaming-cursor" aria-hidden="true">
          {cursorChar}
        </span>
      )}
    </div>
  );
}

function FormattedContent({ text }) {
  if (!text) return null;

  // Split into paragraphs / markdown chunks
  const paragraphs = text.split('\n\n');

  return (
    <>
      {paragraphs.map((para, idx) => {
        if (para.startsWith('### ')) {
          return (
            <h4 key={idx} className="bubble-heading">
              {para.replace('### ', '')}
            </h4>
          );
        }
        if (para.startsWith('- ')) {
          const items = para.split('\n').filter(Boolean);
          return (
            <ul key={idx} className="bubble-list">
              {items.map((it, i) => (
                <li
                  key={i}
                  dangerouslySetInnerHTML={{
                    __html: it.replace(/^- /, '').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'),
                  }}
                />
              ))}
            </ul>
          );
        }
        if (/^\d+\. /.test(para)) {
          const items = para.split('\n').filter(Boolean);
          return (
            <ol key={idx} className="bubble-list">
              {items.map((it, i) => (
                <li
                  key={i}
                  dangerouslySetInnerHTML={{
                    __html: it.replace(/^\d+\. /, '').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'),
                  }}
                />
              ))}
            </ol>
          );
        }
        return (
          <p
            key={idx}
            dangerouslySetInnerHTML={{
              __html: para.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'),
            }}
          />
        );
      })}
    </>
  );
}

export default StreamingText;
