import { Drawer } from "./Drawer";
import { WhatIfSimulator } from "./WhatIfSimulator";

interface Props {
  onClose: () => void;
}

// The right column is now reserved for the ticker list, so the what-if
// simulator (previously always visible in terminal-right) moves into a
// header-triggered drawer instead.
export function WhatIfDrawer({ onClose }: Props) {
  return (
    <Drawer title="What-if event simulator" onClose={onClose}>
      <WhatIfSimulator />
    </Drawer>
  );
}
