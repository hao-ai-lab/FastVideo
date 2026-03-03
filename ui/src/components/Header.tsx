'use client';

import { Fragment } from "react";
import Image from "next/image";
import headerStyles from "@styles/Header.module.css";
import { useActiveTab, useHeaderTitle } from "@/contexts/ActiveTabContext";

export default function Header() {
  const { headerActions } = useActiveTab();
  const title = useHeaderTitle();

  return (
    <header className={headerStyles.header}>
      <Image
        src="/logo.svg"
        alt="FastVideo Logo"
        width={252}
        height={105}
        className={headerStyles.logo}
      />
      <h1 className={headerStyles.title}>{title}</h1>
      {headerActions.length > 0 && (
        <div className={headerStyles.actions}>
          {headerActions.map((action, i) => (
            <Fragment key={i}>{action}</Fragment>
          ))}
        </div>
      )}
    </header>
  );
}
