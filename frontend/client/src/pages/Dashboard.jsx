import {
  AlertTriangle,
  Bell,
  Flag,
  Users,
  Clock,
  Megaphone,
  Activity,
  Wind,
  Thermometer,
  Droplets,
  MapPin,
  Truck,
  Plane,
  Shield,
  Flame,
  HeartHandshake,
  Radio,
  Smartphone,
  Home,
  BarChart3,
  Maximize2,
  ExternalLink,
} from "lucide-react";

import Layout from "../components/Layout";
import "../App.css";

const officialUpdates = [
  {
    agency: "CFA",
    title: "East Gippsland Fire Warning Upgraded",
    text: "Emergency Warning issued for communities in East Gippsland.",
    time: "6 min ago",
    type: "CRITICAL",
    color: "critical",
  },
  {
    agency: "VIC",
    title: "Total Fire Ban Declared",
    text: "Total Fire Ban in effect for Central, North Central, and Mallee districts until 11:59 PM.",
    time: "26 min ago",
    type: "WARNING",
    color: "warning",
  },
  {
    agency: "VicEmergency",
    title: "Smoke conditions worsening near Grampians",
    text: "Smoke levels increasing due to easterly winds.",
    time: "1 hr ago",
    type: "ADVISORY",
    color: "advisory",
  },
];

const resources = [
  {
    icon: Truck,
    name: "Fire Trucks",
    percent: "65%",
    value: "45 deployed / 23 available",
    status: "critical",
  },
  {
    icon: Users,
    name: "Personnel",
    percent: "67%",
    value: "312 deployed / 156 available",
    status: "warning",
  },
  {
    icon: Plane,
    name: "Water Bombers",
    percent: "73%",
    value: "8 deployed / 3 available",
    status: "critical",
  },
  {
    icon: Droplets,
    name: "Water Tankers",
    percent: "65%",
    value: "28 deployed / 15 available",
    status: "safe",
  },
];

const adviceCards = [
  {
    icon: Home,
    title: "Home fire prevention",
    text: "Reduce risk and protect your property.",
    image:
      "https://images.unsplash.com/photo-1523413651479-597eb2da0ad6?auto=format&fit=crop&w=900&q=80",
  },
  {
    icon: MapPin,
    title: "Find emergency services near you",
    text: "Locate hospitals, relief centres, and evacuation points.",
    image:
      "https://images.unsplash.com/photo-1587745416684-47953f16f02f?auto=format&fit=crop&w=900&q=80",
  },
  {
    icon: Smartphone,
    title: "Mobile phone safety warnings",
    text: "Stay informed and avoid network congestion.",
    image:
      "https://images.unsplash.com/photo-1516321497487-e288fb19713f?auto=format&fit=crop&w=900&q=80",
  },
  {
    icon: HeartHandshake,
    title: "Recovery support after an emergency",
    text: "Access support services and community resources.",
    image:
      "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?auto=format&fit=crop&w=900&q=80",
  },
];

export default function Dashboard() {
  return (
    <Layout title="Dashboard">
      <main className="ff-dashboard">
        <section className="ff-hero">
          <div className="ff-hero-overlay"></div>
          <div className="ff-fire-particles">
            <span className="particle p-1"></span>
            <span className="particle p-2"></span>
            <span className="particle p-3"></span>
            <span className="particle p-4"></span>
            <span className="particle p-5"></span>
            <span className="particle p-6"></span>
            <span className="particle p-7"></span>
            <span className="particle p-8"></span>
          </div>

          <div className="ff-hero-content">
            <div>
              <h1>FireFusion Emergency Intelligence Dashboard</h1>
              <p>
                AI-powered bushfire forecasting and misinformation monitoring
                interface for Victoria.
              </p>
            </div>

            <div className="ff-summary-grid">
              <SummaryCard
                icon={AlertTriangle}
                label="Current Risk"
                value="Extreme"
                tone="danger"
              />
              <SummaryCard
                icon={Bell}
                label="Active Alerts"
                value="31"
                tone="red"
              />
              <SummaryCard
                icon={Flag}
                label="Misinformation Flags"
                value="14"
                tone="purple"
              />
              <SummaryCard
                icon={Users}
                label="Resources Deployed"
                value="65%"
                tone="green"
              />
              <SummaryCard
                icon={Clock}
                label="Last Updated"
                value="14:30"
                tone="blue"
              />
            </div>
          </div>
        </section>

        <section className="ff-top-grid">
          <Panel className="ff-updates-panel">
            <PanelHeader
              icon={Megaphone}
              title="Latest Official Updates"
              action="View All"
            />

            <div className="ff-update-list">
              {officialUpdates.map((item) => (
                <UpdateCard key={item.title} {...item} />
              ))}
            </div>
          </Panel>

          <Panel className="ff-incident-panel">
            <PanelHeader icon={Activity} title="Incident Overview" />

            <div className="ff-risk-header">
              <span>Current Risk Level</span>
              <strong>EXTREME</strong>
            </div>

            <div className="ff-risk-meter">
              <span></span>
            </div>

            <div className="ff-metric-grid">
              <Metric icon={Wind} title="Wind Speed" value="45 km/h NW" />
              <Metric icon={Thermometer} title="Temperature" value="41°C" />
              <Metric icon={Droplets} title="Humidity" value="12%" />
              <Metric icon={Users} title="Evacuation Status" value="Active (2 zones)" />
            </div>

            <div className="ff-source-row">
              <span>Data sources: BoM, CFA, VicEmergency</span>
              <b>Updated: 14:30</b>
            </div>
          </Panel>

          <Panel className="ff-resource-panel">
            <PanelHeader icon={Truck} title="Resource Allocation" action="View All" />

            <div className="ff-resource-list">
              {resources.map((item) => (
                <ResourceRow key={item.name} {...item} />
              ))}
            </div>

            <div className="ff-legend">
              <span><i className="safe"></i>Optimal</span>
              <span><i className="warning"></i>Stretched</span>
              <span><i className="critical"></i>Critical</span>
            </div>
          </Panel>
        </section>

        <section className="ff-middle-grid">
          <div>
            <Panel className="ff-map-panel">
              <div className="ff-map-head">
                <h3>Victoria Fire Risk Overview</h3>
                <button>
                  <Maximize2 size={16} />
                </button>
              </div>

              <div className="ff-map-area">
                <div className="ff-map-legend">
                  <span><i className="extreme"></i>Extreme</span>
                  <span><i className="high"></i>High</span>
                  <span><i className="moderate"></i>Moderate</span>
                  <span><i className="low"></i>Low</span>
                  <span><i className="none"></i>No Data</span>
                </div>

                <div className="ff-victoria-map">
                  <div className="zone zone-1">Grampians<br /><small>High</small></div>
                  <div className="zone zone-2">Dandenong Ranges<br /><small>Moderate</small></div>
                  <div className="zone zone-3">Latrobe Valley<br /><small>Moderate</small></div>
                  <div className="zone zone-4">East Gippsland<br /><small>Extreme</small></div>
                </div>

                <button className="ff-map-btn">View Full Map</button>
              </div>
            </Panel>

            <Panel className="ff-chart-panel">
              <div className="ff-panel-title-row">
                <h3>Fire Risk Trend (Next 24 Hours)</h3>
                <a>View Full Analytics</a>
              </div>

              <div className="ff-trend-chart">
                <div className="ff-y-labels">
                  <span>Extreme</span>
                  <span>High</span>
                  <span>Moderate</span>
                  <span>Low</span>
                </div>

                <svg viewBox="0 0 700 170" preserveAspectRatio="none">
                  <polyline
                    points="0,95 45,85 85,55 120,70 160,45 205,35 250,22 300,42 345,36 390,65 440,80 490,110 535,120 575,95 620,100 660,135 700,150"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="5"
                    strokeLinecap="round"
                  />
                </svg>

                <div className="ff-x-labels">
                  <span>15:00</span>
                  <span>18:00</span>
                  <span>21:00</span>
                  <span>00:00</span>
                  <span>03:00</span>
                  <span>06:00</span>
                  <span>09:00</span>
                  <span>12:00</span>
                  <span>15:00</span>
                </div>
              </div>
            </Panel>
          </div>

          <div>
            <Panel className="ff-decision-panel">
              <PanelHeader icon={Shield} title="Decision Support" />

              <div className="ff-decision-grid">
                <DecisionCard
                  icon={Users}
                  title="Evacuation Priority"
                  text="2 zones require immediate review"
                  button="Review Zones"
                  tone="red"
                />
                <DecisionCard
                  icon={Flame}
                  title="Resource Gap"
                  text="Water bombers critically low"
                  button="View Resources"
                  tone="orange"
                />
                <DecisionCard
                  icon={Shield}
                  title="Misinformation Risk"
                  text="14 posts require human review"
                  button="Review Posts"
                  tone="purple"
                />
              </div>
            </Panel>

            <Panel className="ff-advice-panel">
              <PanelHeader icon={Radio} title="Emergency Advice" action="View All" />

              <div className="ff-advice-grid">
                {adviceCards.map((card) => (
                  <AdviceCard key={card.title} {...card} />
                ))}
              </div>
            </Panel>

            <Panel className="ff-misinfo-panel">
              <div className="ff-panel-title-row">
                <h3>Misinformation Posts by Platform</h3>
                <a>View Full Analytics</a>
              </div>

              <div className="ff-misinfo-content">
                <div className="ff-donut">
                  <span>Total<br /><b>378</b></span>
                </div>

                <div className="ff-platform-list">
                  <Platform name="Facebook" percent="38%" count="143" />
                  <Platform name="X (Twitter)" percent="28%" count="106" />
                  <Platform name="Instagram" percent="18%" count="68" />
                  <Platform name="TikTok" percent="10%" count="38" />
                  <Platform name="Other" percent="6%" count="23" />
                </div>

                <div className="ff-theme-box">
                  <h4>Top Misinformation Themes</h4>
                  <Theme label="False evacuation orders" value="42%" width="84%" />
                  <Theme label="Fake fire locations" value="28%" width="58%" />
                  <Theme label="Resource misinformation" value="18%" width="42%" />
                  <Theme label="Other" value="12%" width="30%" />
                </div>
              </div>
            </Panel>
          </div>
        </section>
      </main>
    </Layout>
  );
}

function Panel({ children, className = "" }) {
  return <section className={`ff-panel ${className}`}>{children}</section>;
}

function PanelHeader({ icon: Icon, title, action }) {
  return (
    <div className="ff-panel-header">
      <h3>
        <Icon size={18} />
        {title}
      </h3>
      {action && <a>{action}</a>}
    </div>
  );
}

function SummaryCard({ icon: Icon, label, value, tone }) {
  return (
    <article className="ff-summary-card">
      <Icon size={26} />
      <div>
        <span>{label}</span>
        <strong className={tone}>{value}</strong>
      </div>
    </article>
  );
}

function UpdateCard({ agency, title, text, time, type, color }) {
  return (
    <article className={`ff-update-card ${color}`}>
      <div className="ff-agency">{agency}</div>

      <div className="ff-update-body">
        <div>
          <h4>{title}</h4>
          <p>{text}</p>
          <small>{time}</small>
        </div>

        <div>
          <span>{type}</span>
          <button>View Details</button>
        </div>
      </div>
    </article>
  );
}

function Metric({ icon: Icon, title, value }) {
  return (
    <article className="ff-metric-card">
      <Icon size={28} />
      <div>
        <span>{title}</span>
        <strong>{value}</strong>
      </div>
    </article>
  );
}

function ResourceRow({ icon: Icon, name, percent, value, status }) {
  return (
    <article className="ff-resource-row">
      <div className="ff-resource-info">
        <span>
          <Icon size={17} />
          {name}
        </span>
        <b>{percent}</b>
        <small>{value}</small>
      </div>

      <div className="ff-resource-bar">
        <i className={status} style={{ width: percent }}></i>
      </div>
    </article>
  );
}

function DecisionCard({ icon: Icon, title, text, button, tone }) {
  return (
    <article className={`ff-decision-card ${tone}`}>
      <Icon size={32} />
      <div>
        <h4>{title}</h4>
        <p>{text}</p>
        <button>
          <ExternalLink size={14} />
          {button}
        </button>
      </div>
    </article>
  );
}

function AdviceCard({ icon: Icon, title, text, image }) {
  return (
    <article className="ff-advice-card">
      <img src={image} alt={title} />
      <div className="ff-advice-cover"></div>
      <div className="ff-advice-text">
        <Icon size={24} />
        <h4>{title}</h4>
        <p>{text}</p>
      </div>
    </article>
  );
}

function Platform({ name, percent, count }) {
  return (
    <div className="ff-platform-row">
      <span>{name}</span>
      <b>{percent}</b>
      <small>{count}</small>
    </div>
  );
}

function Theme({ label, value, width }) {
  return (
    <div className="ff-theme-row">
      <span>{label}</span>
      <div>
        <i style={{ width }}></i>
      </div>
      <b>{value}</b>
    </div>
  );
}