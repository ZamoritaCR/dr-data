-- Dr. Data V2: Correction Learning + Session Persistence
-- Run in Supabase SQL Editor for project wkfewpynskakgbetscsa

CREATE TABLE IF NOT EXISTS drdata_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT UNIQUE NOT NULL,
  twbx_filename TEXT,
  tableau_spec JSONB,
  pipeline_state JSONB,
  current_stage INTEGER DEFAULT 0,
  config JSONB,
  data_profile JSONB,
  translations JSONB,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON drdata_sessions(session_id);

CREATE TABLE IF NOT EXISTS drdata_corrections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  field_path TEXT NOT NULL,
  original_value JSONB NOT NULL,
  corrected_value JSONB NOT NULL,
  correction_type TEXT NOT NULL,
  worksheet_name TEXT,
  mark_type TEXT,
  context JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_corrections_stage ON drdata_corrections(stage);
CREATE INDEX IF NOT EXISTS idx_corrections_type ON drdata_corrections(correction_type);
CREATE INDEX IF NOT EXISTS idx_corrections_mark ON drdata_corrections(mark_type);
