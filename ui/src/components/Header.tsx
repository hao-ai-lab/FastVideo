'use client';

import Image from "next/image";
import headerStyles from "@styles/Header.module.css";
import CreateJobButton from "./CreateJobButton";

export default function Header() {
  return (
    <header className={headerStyles.header}>
      <Image
        src="/logo.svg"
        alt="FastVideo Logo"
        width={252}
        height={105}
        className={headerStyles.logo}
      />
      <h1 className={headerStyles.title}>Jobs</h1>
      <CreateJobButton />
    </header>
  );
}
